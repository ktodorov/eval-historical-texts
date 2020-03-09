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


class NERRNNModel(ModelBase):
    def __init__(
            self,
            arguments_service: NERArgumentsService,
            data_service: DataService,
            metrics_service: MetricsService):
        super().__init__(data_service)

        self._include_pretrained = arguments_service.include_pretrained_model
        additional_size = arguments_service.pretrained_model_size if self._include_pretrained else 0
        self._learn_embeddings = arguments_service.learn_new_embeddings

        self._metrics_service = metrics_service
        self._metric_types = arguments_service.metric_types
        number_of_tags = 11 if arguments_service.label_type == NERType.Coarse else 38

        # maps each token to an embedding_dim vector
        lstm_input_size = additional_size
        if self._learn_embeddings:
            self.embedding = nn.Embedding(arguments_service.pretrained_vocabulary_size, arguments_service.embeddings_size)
            self.dropout = nn.Dropout(arguments_service.dropout)
            lstm_input_size += arguments_service.embeddings_size

        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(lstm_input_size, arguments_service.hidden_dimension, batch_first=True, bidirectional=True)

        # fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(arguments_service.hidden_dimension * 2, number_of_tags)

    def forward(self, input_batch, debug=False, **kwargs):
        sequences, targets, lengths, pretrained_representations = input_batch

        # apply the embedding layer that maps each token to its embedding
        # dim: batch_size x batch_max_len x embedding_dim
        if self._learn_embeddings:
            embedded = self.dropout(self.embedding(sequences))

            if self._include_pretrained:
                embedded = torch.cat((embedded, pretrained_representations), dim=2)
        else:
            embedded = pretrained_representations

        x_packed = pack_padded_sequence(embedded, lengths, batch_first=True)

        # run the LSTM along the sentences of length batch_max_len
        # dim: batch_size x batch_max_len x lstm_hidden_dim
        packed_output, _ = self.lstm(x_packed)

        rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # reshape the Variable so that each row contains one token
        # dim: batch_size*batch_max_len x lstm_hidden_dim
        rnn_output = rnn_output.reshape(-1, rnn_output.shape[2])

        # apply the fully connected layer and obtain the output for each token
        # dim: batch_size*batch_max_len x num_tags
        output = self.fc(rnn_output)

        # dim: batch_size*batch_max_len x num_tags
        return F.log_softmax(output, dim=1), targets

    def calculate_accuracies(self, batch, outputs, output_characters=False) -> Dict[MetricType, float]:
        output, targets = outputs
        predictions = output.max(dim=1)[1].cpu().detach().numpy()

        targets = targets.reshape(-1).cpu().detach().numpy()

        mask = np.array((targets != -1), dtype=bool)
        predicted_labels = predictions[mask]
        target_labels = targets[mask]

        metrics = {}

        if MetricType.F1Score in self._metric_types:
            f1_score = self._metrics_service.calculate_f1_score(
                predicted_labels, target_labels)
            metrics[MetricType.F1Score] = f1_score

        return metrics, None

    def compare_metric(self, best_metric: Metric, new_metric: Metric) -> bool:
        if best_metric.is_new:
            return True

        return best_metric.get_accuracy_metric(MetricType.F1Score) <= new_metric.get_accuracy_metric(MetricType.F1Score)
