import os
from typing import Callable

from transformers import BertForMaskedLM, BertPreTrainedModel, BertConfig

from entities.model_checkpoint import ModelCheckpoint
from entities.metric import Metric
from models.model_base import ModelBase

from services.semantic_arguments_service import SemanticArgumentsService
from services.data_service import DataService


class KBertModel(ModelBase):
    def __init__(
            self,
            arguments_service: SemanticArgumentsService,
            data_service: DataService,
            output_hidden_states: bool = False):
        super(KBertModel, self).__init__(data_service)

        self._output_hidden_states = output_hidden_states

        if arguments_service.resume_training or arguments_service.evaluate or arguments_service.run_experiments:
            self._bert_model = None
        else:
            config = BertConfig.from_pretrained(arguments_service.pretrained_weights)
            config.output_hidden_states = output_hidden_states

            self._bert_model: BertPreTrainedModel = self._model_type.from_pretrained(
                arguments_service.pretrained_weights, config=config)

        self._arguments_service = arguments_service

    def forward(self, input_batch, **kwargs):
        if isinstance(input_batch, tuple):
            (inputs, labels) = input_batch
            outputs = self._bert_model.forward(inputs, masked_lm_labels=labels)
            return outputs[0]
        else:
            outputs = self._bert_model.forward(input_batch)
            return outputs[1][-1]

    def named_parameters(self):
        return self._bert_model.named_parameters()

    def parameters(self):
        return self._bert_model.parameters()

    def compare_metric(self, best_metric: Metric, new_metrics: Metric) -> bool:
        if best_metric.is_new or best_metric.get_current_loss() > new_metrics.get_current_loss():
            return True

        return False

    def save(
            self,
            path: str,
            epoch: int,
            iteration: int,
            best_metrics: object,
            name_prefix: str = None) -> bool:

        checkpoint_name = self._arguments_service.checkpoint_name

        if checkpoint_name:
            name_prefix = f'{name_prefix}_{checkpoint_name}'

        saved = super().save(path, epoch, iteration, best_metrics,
                             name_prefix, save_model_dict=False)

        if not saved:
            return saved

        pretrained_weights_path = self._get_pretrained_path(
            path, name_prefix, create_if_missing=True)

        self._bert_model.save_pretrained(pretrained_weights_path)

        return saved

    def load(
            self,
            path: str,
            name_prefix: str = None,
            load_model_dict: bool = True,
            load_model_only: bool = False) -> ModelCheckpoint:

        checkpoint_name = self._arguments_service.checkpoint_name

        if checkpoint_name:
            name_prefix = f'{name_prefix}_{checkpoint_name}'

        model_checkpoint = super().load(path, name_prefix, load_model_dict=False)
        if not load_model_only and not model_checkpoint:
            return None

        if load_model_dict:
            self._load_kbert_model(path, name_prefix)

        return model_checkpoint

    @property
    def _model_type(self) -> BertPreTrainedModel:
        return BertForMaskedLM

    def _load_kbert_model(self, path: str, name_prefix: str):
        pretrained_weights_path = self._get_pretrained_path(path, name_prefix)

        config = BertConfig.from_pretrained(pretrained_weights_path)
        config.output_hidden_states = True

        self._bert_model = self._model_type.from_pretrained(
            pretrained_weights_path, config=config).to(self._arguments_service.device)

    def _get_pretrained_path(self, path: str, name_prefix: str, create_if_missing: bool = False):
        file_name = f'{name_prefix}_pretrained_weights'
        pretrained_weights_path = os.path.join(path, file_name)

        if create_if_missing and not os.path.exists(pretrained_weights_path):
            os.mkdir(pretrained_weights_path)

        return pretrained_weights_path