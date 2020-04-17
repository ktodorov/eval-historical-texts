import torch
import torch.nn as nn

import numpy as np

from models.ner_rnn.rnn_encoder import RNNEncoder
from models.ner_rnn.linear_crf import LinearCRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Dict, List
from overrides import overrides

from entities.metric import Metric
from enums.metric_type import MetricType

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

        self.device = arguments_service.device
        number_of_tags = process_service.get_labels_amount()
        self._metric_types = arguments_service.metric_types

        self.rnn_encoder = RNNEncoder(
            number_of_tags=number_of_tags + 2,  # TODO: Check if + 2 is valid
            use_attention=arguments_service.use_attention,
            include_pretrained_model=arguments_service.include_pretrained_model,
            pretrained_model_size=arguments_service.pretrained_model_size,
            learn_new_embeddings=arguments_service.learn_new_embeddings,
            vocabulary_size=tokenizer_service.vocabulary_size,
            embeddings_size=arguments_service.embeddings_size,
            dropout=arguments_service.dropout,
            hidden_dimension=arguments_service.hidden_dimension,
            bidirectional=arguments_service.bidirectional_rnn)

        self.crf_layer = LinearCRF(
            num_of_tags=number_of_tags,
            device=arguments_service.device)

    @overrides
    def forward(self, input_batch):
        sequences, targets, lengths, pretrained_representations = input_batch

        rnn_outputs = self.rnn_encoder.forward(
            sequences=sequences,
            lengths=lengths,
            pretrained_representations=pretrained_representations)

        mask, targets = self._create_mask(
            inputs=sequences,
            targets=targets,
            rnn_outputs=rnn_outputs)

        loss = self.crf_layer.neg_log_likelihood_loss(
            rnn_outputs,
            mask,
            targets)

        _, predictions = self.crf_layer.forward(
            rnn_outputs,
            mask)

        return predictions, loss

    @overrides
    def calculate_accuracies(self, batch, outputs, output_characters=False) -> Dict[MetricType, float]:
        output, _ = outputs
        _, targets, _, _ = batch

        predictions = output.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        mask = np.array((targets != -1), dtype=bool)
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

        return best_metric.get_accuracy_metric(MetricType.F1Score) <= new_metric.get_accuracy_metric(MetricType.F1Score)

    def _create_mask(self, inputs, rnn_outputs, targets=None):
        mask = inputs > -1
        mask = mask[:, :rnn_outputs.size(1)]
        if self.training:
            targets = targets[:, :rnn_outputs.size(1)]

        return mask, targets
