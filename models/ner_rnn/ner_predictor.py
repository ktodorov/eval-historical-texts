import torch
import torch.nn as nn

import numpy as np

from models.ner_rnn.rnn_encoder import RNNEncoder
from models.ner_rnn.linear_crf import LinearCRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Dict, List
from overrides import overrides

from entities.metric import Metric
from entities.data_output_log import DataOutputLog

from enums.metric_type import MetricType
from enums.tag_measure_averaging import TagMeasureAveraging
from enums.tag_measure_type import TagMeasureType

from models.model_base import ModelBase

from services.arguments.ner_arguments_service import NERArgumentsService
from services.data_service import DataService
from services.metrics_service import MetricsService
from services.tokenizer_service import TokenizerService
from services.process.ner_process_service import NERProcessService


class NERPredictor(ModelBase):

    def __init__(
            self,
            arguments_service: NERArgumentsService,
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
            number_of_tags=self.number_of_tags,
            use_attention=arguments_service.use_attention,
            include_pretrained_model=arguments_service.include_pretrained_model,
            pretrained_model_size=arguments_service.pretrained_model_size,
            learn_new_embeddings=arguments_service.learn_new_embeddings,
            vocabulary_size=tokenizer_service.vocabulary_size,
            embeddings_size=arguments_service.embeddings_size,
            dropout=arguments_service.dropout,
            hidden_dimension=arguments_service.hidden_dimension,
            bidirectional=arguments_service.bidirectional_rnn,
            number_of_layers=arguments_service.number_of_layers)

        self.pad_idx = self._process_service.get_entity_label(
            self._process_service.PAD_TOKEN)

        self.crf_layer = LinearCRF(
            num_of_tags=self.number_of_tags,
            device=arguments_service.device,
            start_tag=process_service.get_entity_label(
                process_service.START_TOKEN),
            stop_tag=process_service.get_entity_label(
                process_service.STOP_TOKEN),
            pad_tag=self.pad_idx)

        self.tag_measure_averages = [
            TagMeasureAveraging.Macro, TagMeasureAveraging.Micro, TagMeasureAveraging.Weighted]
        self.tag_measure_types = [TagMeasureType.Strict]

    @overrides
    def forward(self, input_batch):
        sequences, targets, lengths, pretrained_representations, position_changes = input_batch

        mask = self._create_mask(sequences)

        rnn_outputs = self.rnn_encoder.forward(
            sequences=sequences,
            lengths=lengths,
            pretrained_representations=pretrained_representations,
            mask=mask)

        rnn_outputs, targets, mask = self._restore_position_changes(
            position_changes,
            rnn_outputs,
            targets,
            mask)

        loss = self.crf_layer.neg_log_likelihood(
            rnn_outputs,
            targets,
            mask)

        predictions = self.crf_layer.forward(
            rnn_outputs,
            mask)

        return predictions, loss, targets

    @overrides
    def calculate_accuracies(self, batch, outputs, output_characters=False) -> Dict[MetricType, float]:
        output, _, targets = outputs
        # _, targets, _, _, _ = batch

        predicted_labels = np.array(output)
        targets = targets.cpu().detach().numpy()

        mask = np.array(
            (targets != self._process_service.get_entity_label(self._process_service.PAD_TOKEN)), dtype=bool)
        predicted_labels = np.hstack(predicted_labels)
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
            batch_size = targets.shape[0]
            for i in range(batch_size):
                predicted_string = ','.join(
                    [str(predicted_label) for predicted_label in output[i]])
                target_string = ','.join([str(target_label)
                                          for target_label in targets[i] if target_label != self.pad_idx])

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

    def _create_mask(self, input_sequences: torch.Tensor):
        mask = input_sequences.ne(self.pad_idx).float()
        return mask

    def _restore_position_changes(
            self,
            position_changes,
            rnn_outputs,
            targets,
            mask):
        if position_changes is None:
            return rnn_outputs, targets, mask

        batch_size, sequence_length, _ = rnn_outputs.shape
        new_rnn_outputs = torch.zeros(
            (batch_size, sequence_length, self.number_of_tags), dtype=rnn_outputs.dtype).to(self.device)
        new_targets = torch.zeros((batch_size, sequence_length), dtype=targets.dtype).to(self.device)
        new_mask = torch.zeros((batch_size, sequence_length), dtype=mask.dtype).to(self.device)

        for i, current_position_changes in enumerate(position_changes):
            if current_position_changes is None:
                new_rnn_outputs[i] = rnn_outputs[i]
                new_targets[i] = targets[i]
                new_mask[i] = mask[i]
                continue

            for old_position, new_positions in current_position_changes.items():
                new_rnn_outputs[i,old_position,:] = torch.mean(rnn_outputs[i, new_positions], dim=0)
                new_targets[i,old_position] = targets[i, new_positions[0]]
                new_mask[i,old_position] = torch.max(mask[i, new_positions])

        return new_rnn_outputs, new_targets, new_mask
