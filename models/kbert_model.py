import os

from transformers import BertForMaskedLM

from entities.model_checkpoint import ModelCheckpoint
from models.model_base import ModelBase

from services.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService


class KBertModel(ModelBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            data_service: DataService):
        super(KBertModel, self).__init__(data_service, arguments_service)

        self._bert_model = BertForMaskedLM.from_pretrained(
            arguments_service.get_argument('pretrained_weights'))

    def forward(self, input_batch, **kwargs):
        (inputs, labels) = input_batch
        outputs = self._bert_model.forward(inputs, masked_lm_labels=labels)
        return outputs

    def named_parameters(self):
        return self._bert_model.named_parameters()

    def parameters(self):
        return self._bert_model.parameters()

    def calculate_accuracy(self, predictions, targets) -> int:
        return 0

    def compare_metric(self, best_metric, metrics) -> bool:
        if best_metric is None or best_metric > metrics:
            return True

    def save(
            self,
            path: str,
            epoch: int,
            iteration: int,
            best_metrics: object,
            name_prefix: str = None) -> bool:

        saved = super().save(path, epoch, iteration, best_metrics,
                             name_prefix, save_model_dict=False)

        if not saved:
            return saved

        pretrained_weights_path = self._get_pretrained_path(
            path, name_prefix, create_if_missing=True)

        self._bert_model.save_pretrained(pretrained_weights_path)

        return saved

    def load(self, path: str, name_prefix: str = None) -> ModelCheckpoint:
        model_checkpoint = super().load(path, name_prefix, load_model_dict=False)
        if not model_checkpoint:
            return None

        pretrained_weights_path = self._get_pretrained_path(path, name_prefix)

        self._bert_model = BertForMaskedLM.from_pretrained(
            pretrained_weights_path).to(self._arguments_service.get_argument('device'))

        return model_checkpoint

    def _get_pretrained_path(self, path: str, name_prefix: str, create_if_missing: bool = False):
        pretrained_weights_path = os.path.join(
            path, f'{name_prefix}_pretrained_weights')

        if create_if_missing and not os.path.exists(pretrained_weights_path):
            os.mkdir(pretrained_weights_path)

        return pretrained_weights_path
