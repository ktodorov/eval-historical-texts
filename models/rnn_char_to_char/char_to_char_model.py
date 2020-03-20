import numpy as np

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from typing import Dict

from enums.metric_type import MetricType

from entities.metric import Metric

from models.model_base import ModelBase

from services.arguments.postocr_characters_arguments_service import PostOCRCharactersArgumentsService
from services.data_service import DataService
from services.metrics_service import MetricsService
from services.vocabulary_service import VocabularyService


class CharToCharModel(ModelBase):
    def __init__(
            self,
            arguments_service: PostOCRCharactersArgumentsService,
            vocabulary_service: VocabularyService,
            data_service: DataService,
            metrics_service: MetricsService):
        super().__init__(data_service)

        self._vocabulary_service = vocabulary_service
        self._metrics_service = metrics_service
        self._arguments_service = arguments_service

        self._include_pretrained = arguments_service.include_pretrained_model
        self._learn_embeddings = arguments_service.learn_new_embeddings

        self._metric_types = arguments_service.metric_types

        # maps each token to an embedding_dim vector
        additional_size = arguments_service.pretrained_model_size if self._include_pretrained else 0
        lstm_input_size = additional_size
        if self._learn_embeddings:
            self.embedding = nn.Embedding(
                vocabulary_service.vocabulary_size(), arguments_service.embeddings_size)
            self.dropout = nn.Dropout(arguments_service.dropout)
            lstm_input_size += arguments_service.embeddings_size

        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(
            lstm_input_size, arguments_service.hidden_dimension, batch_first=True, bidirectional=arguments_service.bidirectional)

        multiplication_factor = 2 if arguments_service.bidirectional else 1
        self.fc = nn.Linear(arguments_service.hidden_dimension *
                            multiplication_factor, vocabulary_service.vocabulary_size())

    def forward(self, input_batch, debug=False, **kwargs):
        sequences, targets, lengths, pretrained_representations, offset_lists = input_batch

        # apply the embedding layer that maps each token to its embedding
        # dim: batch_size x batch_max_len x embedding_dim
        if self._learn_embeddings:
            embedded = self.dropout(self.embedding(sequences))

            if self._include_pretrained:
                embedded = self._add_pretrained_information(
                    embedded, pretrained_representations, offset_lists)
        else:
            embedded = pretrained_representations

        x_packed = pack_padded_sequence(embedded, lengths, batch_first=True)

        # run the LSTM along the sentences of length batch_max_len
        # dim: batch_size x batch_max_len x lstm_hidden_dim
        packed_output, _ = self.lstm.forward(x_packed)

        rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # # reshape the Variable so that each row contains one token
        # # dim: batch_size*batch_max_len x lstm_hidden_dim
        # rnn_output = rnn_output.reshape(-1, rnn_output.shape[2])

        output = self.fc.forward(rnn_output)

        # dim: batch_size*batch_max_len x num_tags

        if output.shape[1] < targets.shape[1]:
            padded_output = torch.zeros((output.shape[0], targets.shape[1], output.shape[2])).to(
                self._arguments_service.device)
            padded_output[:, :output.shape[1], :] = output
            output = padded_output
        elif output.shape[1] > targets.shape[1]:
            padded_targets = torch.zeros((targets.shape[0], output.shape[1]), dtype=torch.int64).to(
                self._arguments_service.device)
            padded_targets[:, :targets.shape[1]] = targets
            targets = padded_targets

        return output, targets

    def calculate_accuracies(self, batch, outputs, output_characters=False) -> Dict[MetricType, float]:
        output, targets = outputs
        output_dim = output.shape[-1]
        predictions = output.max(dim=2)[1].cpu().detach().numpy()

        targets = targets.cpu().detach().numpy()
        predicted_characters = []
        target_characters = []

        for i in range(targets.shape[0]):
            indices = np.array(
                (targets[i] != self._vocabulary_service.pad_token), dtype=bool)
            predicted_characters.append(predictions[i][indices])
            target_characters.append(targets[i][indices])

        metrics = {}

        if MetricType.JaccardSimilarity in self._metric_types:
            predicted_tokens = [self._vocabulary_service.ids_to_string(
                x) for x in predicted_characters]
            target_tokens = [self._vocabulary_service.ids_to_string(
                x) for x in target_characters]
            jaccard_score = np.mean([self._metrics_service.calculate_jaccard_similarity(
                target_tokens[i], predicted_tokens[i]) for i in range(len(predicted_tokens))])

            metrics[MetricType.JaccardSimilarity] = jaccard_score

        character_results = None
        if MetricType.LevenshteinDistance in self._metric_types:
            predicted_strings = [self._vocabulary_service.ids_to_string(
                x) for x in predicted_characters]
            target_strings = [self._vocabulary_service.ids_to_string(
                x) for x in target_characters]

            levenshtein_distance = np.mean([self._metrics_service.calculate_normalized_levenshtein_distance(
                predicted_strings[i], target_strings[i]) for i in range(len(predicted_strings))])

            metrics[MetricType.LevenshteinDistance] = levenshtein_distance

            if output_characters:
                character_results = []
                ocr_texts_tensor, _, lengths, _, _ = batch
                ocr_texts = ocr_texts_tensor.cpu().detach().tolist()
                for i in range(len(ocr_texts)):
                    input_string = self._vocabulary_service.ids_to_string(
                        ocr_texts[i])

                    character_results.append(
                        [input_string, predicted_strings[i], target_strings[i]])

        return metrics, character_results

    def compare_metric(self, best_metric: Metric, new_metric: Metric) -> bool:
        if best_metric.is_new:
            return True

        result = best_metric.get_current_loss() > new_metric.get_current_loss()
        return result

    def _add_pretrained_information(self, embedded, pretrained_representations, offset_lists):
        batch_size = embedded.shape[0]
        pretrained_embedding_size = pretrained_representations.shape[2]

        new_embedded = torch.zeros(
            (batch_size, embedded.shape[1], embedded.shape[2] + pretrained_representations.shape[2])).to(self._arguments_service.device)
        new_embedded[:, :, :embedded.shape[2]] = embedded

        for i in range(batch_size):
            inserted_count = 0
            last_item = 0
            for p_i, offset in enumerate(offset_lists[i]):
                current_offset = 0
                if offset[0] == offset[1]:
                    current_offset = 1

                for k in range(offset[0] + inserted_count, offset[1] + inserted_count + current_offset):
                    if offset[0] < last_item:
                        continue

                    last_item = offset[1]

                    new_embedded[i, k, -pretrained_embedding_size:
                                 ] = pretrained_representations[i, p_i]

                if offset[0] == offset[1]:
                    inserted_count += 1

        return new_embedded
