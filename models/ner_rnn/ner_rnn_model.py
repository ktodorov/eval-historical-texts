import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from typing import List, Dict

from entities.metric import Metric
from enums.metric_type import MetricType
from enums.ner_type import NERType

from models.model_base import ModelBase

from services.arguments.ner_arguments_service import NERArgumentsService
from services.data_service import DataService
from services.metrics_service import MetricsService
from services.tokenizer_service import TokenizerService
from services.process.ner_process_service import NERProcessService


class NERRNNModel(ModelBase):
    def __init__(
            self,
            arguments_service: NERArgumentsService,
            data_service: DataService,
            metrics_service: MetricsService,
            process_service: NERProcessService,
            tokenizer_service: TokenizerService):
        super().__init__(data_service, arguments_service)

        self._include_pretrained = arguments_service.include_pretrained_model
        additional_size = arguments_service.pretrained_model_size if self._include_pretrained else 0
        self._learn_embeddings = arguments_service.learn_new_embeddings

        self._metrics_service = metrics_service
        self._metric_types = arguments_service.metric_types
        number_of_tags = process_service.get_labels_amount()

        # maps each token to an embedding_dim vector
        lstm_input_size = additional_size
        if self._learn_embeddings:
            self.embedding = nn.Embedding(
                tokenizer_service.vocabulary_size, arguments_service.embeddings_size)
            self.dropout = nn.Dropout(arguments_service.dropout)
            lstm_input_size += arguments_service.embeddings_size

        # the LSTM takes embedded sentence
        self.literal_lstm = nn.LSTM(
            lstm_input_size, arguments_service.hidden_dimension, batch_first=True, bidirectional=True)
        self.metonymic_lstm = nn.LSTM(
            lstm_input_size, arguments_service.hidden_dimension, batch_first=True, bidirectional=True)

        # fc layer transforms the output to give the final output layer
        self.literal_output_layer = nn.Linear(
            arguments_service.hidden_dimension * 2, number_of_tags)
        self.metonymic_output_layer = nn.Linear(
            arguments_service.hidden_dimension * 2, number_of_tags)

    def forward(self, input_batch, debug=False, **kwargs):
        sequences, literal_targets, metonymic_targets, lengths, pretrained_representations = input_batch

        # apply the embedding layer that maps each token to its embedding
        # dim: batch_size x batch_max_len x embedding_dim
        if self._learn_embeddings:
            embedded = self.dropout(self.embedding(sequences))

            if self._include_pretrained:
                embedded = torch.cat(
                    (embedded, pretrained_representations), dim=2)
        else:
            embedded = pretrained_representations

        x_packed = pack_padded_sequence(embedded, lengths, batch_first=True)

        # run the LSTM along the sentences of length batch_max_len
        # dim: batch_size x batch_max_len x lstm_hidden_dim
        literal_packed_output, _ = self.literal_lstm(x_packed)
        literal_rnn_output, _ = pad_packed_sequence(
            literal_packed_output, batch_first=True)
        literal_rnn_output = literal_rnn_output.reshape(
            -1, literal_rnn_output.shape[2])
        literal_output = self.literal_output_layer(literal_rnn_output)

        metonymic_packed_output, _ = self.metonymic_lstm(x_packed)
        metonymic_rnn_output, _ = pad_packed_sequence(
            metonymic_packed_output, batch_first=True)
        metonymic_rnn_output = metonymic_rnn_output.reshape(
            -1, metonymic_rnn_output.shape[2])
        metonymic_output = self.metonymic_output_layer(metonymic_rnn_output)

        return (
            F.log_softmax(literal_output, dim=1),
            F.log_softmax(metonymic_output, dim=1),
            literal_targets,
            metonymic_targets
        )

    def calculate_accuracies(self, batch, outputs, output_characters=False) -> Dict[MetricType, float]:
        literal_output, metonymic_output, literal_targets, metonymic_targets = outputs

        metrics = {}
        if MetricType.F1Score in self._metric_types:
            literal_f1_score = self._calculate_f1_score(
                literal_output,
                literal_targets)

            metonymic_f1_score = self._calculate_f1_score(
                metonymic_output,
                metonymic_targets)

            f1_score = (literal_f1_score + metonymic_f1_score) / 2
            metrics[MetricType.F1Score] = f1_score

        return metrics, None

    def _calculate_f1_score(self, output, targets):
        predictions = output.max(dim=1)[1].cpu().detach().numpy()
        targets = targets.reshape(-1).cpu().detach().numpy()

        mask = np.array((targets != -1), dtype=bool)
        predicted_labels = predictions[mask]
        target_labels = targets[mask]

        f1_score = self._metrics_service.calculate_f1_score(
            predicted_labels, target_labels)

        return f1_score

    def compare_metric(self, best_metric: Metric, new_metric: Metric) -> bool:
        if best_metric.is_new:
            return True

        return best_metric.get_accuracy_metric(MetricType.F1Score) <= new_metric.get_accuracy_metric(MetricType.F1Score)
