import torch
import torch.nn as nn

import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Dict, List
from overrides import overrides

from entities.metric import Metric
from entities.data_output_log import DataOutputLog
from entities.batch_representation import BatchRepresentation

from enums.metric_type import MetricType
from enums.tag_measure_averaging import TagMeasureAveraging
from enums.tag_measure_type import TagMeasureType

from models.ner_rnn.rnn_encoder import RNNEncoder
from models.ner_rnn.conditional_random_field import ConditionalRandomField
from models.model_base import ModelBase

from services.arguments.ner_arguments_service import NERArgumentsService
from services.data_service import DataService
from services.metrics_service import MetricsService
from services.tokenizer_service import TokenizerService
from services.pretrained_representations_service import PretrainedRepresentationsService
from services.process.ner_process_service import NERProcessService


class NERPredictor(ModelBase):

    def __init__(
            self,
            arguments_service: NERArgumentsService,
            pretrained_representations_service: PretrainedRepresentationsService,
            data_service: DataService,
            metrics_service: MetricsService,
            process_service: NERProcessService,
            tokenizer_service: TokenizerService):
        super().__init__(data_service, arguments_service)

        self._metrics_service = metrics_service
        self._process_service = process_service

        self.device = arguments_service.device
        self.number_of_tags = process_service.get_labels_amount()
        self._metric_types = arguments_service.metric_types

        self.rnn_encoder = RNNEncoder(
            pretrained_representations_service=pretrained_representations_service,
            device=self.device,
            number_of_tags=self.number_of_tags,
            use_attention=arguments_service.use_attention,
            include_pretrained_model=arguments_service.include_pretrained_model,
            pretrained_model_size=arguments_service.pretrained_model_size,
            include_fasttext_model=arguments_service.include_fasttext_model,
            fasttext_model_size=arguments_service.fasttext_model_size,
            learn_new_embeddings=arguments_service.learn_new_embeddings,
            vocabulary_size=tokenizer_service.vocabulary_size,
            embeddings_size=arguments_service.embeddings_size,
            dropout=arguments_service.dropout,
            hidden_dimension=arguments_service.hidden_dimension,
            bidirectional=arguments_service.bidirectional_rnn,
            number_of_layers=arguments_service.number_of_layers,
            merge_subword_embeddings=arguments_service.merge_subwords,
            learn_character_embeddings=arguments_service.learn_character_embeddings,
            character_embeddings_size=arguments_service.character_embeddings_size)

        self.pad_idx = self._process_service.get_entity_label(
            self._process_service.PAD_TOKEN)
        start_token_id = self._process_service.get_entity_label(
            self._process_service.PAD_TOKEN)
        stop_token_id = self._process_service.get_entity_label(
            self._process_service.PAD_TOKEN)

        self.crf_layer = ConditionalRandomField(
            num_of_tags=self.number_of_tags,
            device=arguments_service.device,
            context_emb=arguments_service.hidden_dimension,
            start_token_id=start_token_id,
            stop_token_id=stop_token_id,
            pad_token_id=self.pad_idx)

        self.tag_measure_averages = [
            TagMeasureAveraging.Macro, TagMeasureAveraging.Micro, TagMeasureAveraging.Weighted]
        self.tag_measure_types = [TagMeasureType.Strict]

    @overrides
    def forward(self, batch_representation: BatchRepresentation):
        rnn_outputs, lengths, targets = self.rnn_encoder.forward(batch_representation)

        mask = self._create_mask(rnn_outputs, lengths)
        loss, predictions = self.crf_layer.forward(rnn_outputs, lengths, targets, mask)

        return predictions, loss, targets, lengths

    @overrides
    def calculate_accuracies(
        self,
        batch: BatchRepresentation,
        outputs,
        output_characters=False) -> Dict[MetricType, float]:
        output, _, targets, lengths = outputs

        predictions = output.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        mask = np.array((targets != self.pad_idx), dtype=bool)
        predicted_labels = predictions[mask]
        target_labels = targets[mask]

        metrics: Dict[MetricType, float] = {}

        for tag_measure_averaging in self.tag_measure_averages:
            for tag_measure_type in self.tag_measure_types:
                (precision_score, recall_score, f1_score, _) = self._metrics_service.calculate_precision_recall_fscore_support(
                    predicted_labels,
                    target_labels,
                    tag_measure_type,
                    tag_measure_averaging)

                if MetricType.F1Score in self._metric_types:
                    key = self._create_measure_key(
                        MetricType.F1Score, tag_measure_averaging, tag_measure_type)
                    metrics[key] = f1_score

                if MetricType.PrecisionScore in self._metric_types:
                    key = self._create_measure_key(
                        MetricType.PrecisionScore, tag_measure_averaging, tag_measure_type)
                    metrics[key] = precision_score

                if MetricType.RecallScore in self._metric_types:
                    key = self._create_measure_key(
                        MetricType.RecallScore, tag_measure_averaging, tag_measure_type)
                    metrics[key] = recall_score

        output_log = None
        if output_characters:
            output_log = DataOutputLog()
            for i in range(batch.batch_size):
                predicted_string = ','.join(
                    [self._process_service.get_entity_by_label(predicted_label) for predicted_label in predictions[i][:lengths[i]]])
                target_string = ','.join([self._process_service.get_entity_by_label(target_label)
                                          for target_label in targets[i][:lengths[i]]])

                output_log.add_new_data(
                    output_data=predicted_string,
                    true_data=target_string)

        return metrics, output_log

    @overrides
    def compare_metric(self, best_metric: Metric, new_metric: Metric) -> bool:
        if best_metric.is_new:
            return True
        key = self._create_measure_key(
            MetricType.F1Score, TagMeasureAveraging.Macro, TagMeasureType.Strict)
        return best_metric.get_accuracy_metric(key) <= new_metric.get_accuracy_metric(key)

    def _create_measure_key(
            self,
            metric_type: MetricType,
            tag_measure_averaging: TagMeasureAveraging,
            tag_measure_type: TagMeasureType):
        key = f'{metric_type.value}-{tag_measure_averaging.value}-{tag_measure_type.value}'
        return key

    def _create_mask(self, rnn_outputs: torch.Tensor, lengths: torch.Tensor):
        batch_size = rnn_outputs.size(0)
        sent_len = rnn_outputs.size(1)
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long).view(
            1, sent_len).expand(batch_size, sent_len).to(self.device)
        mask = torch.le(maskTemp, lengths.view(batch_size, 1).expand(
            batch_size, sent_len)).to(self.device)
        return mask
