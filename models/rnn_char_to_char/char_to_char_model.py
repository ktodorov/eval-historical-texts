import numpy as np

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from typing import Dict

from overrides import overrides

from enums.metric_type import MetricType
from enums.embedding_type import EmbeddingType

from entities.metric import Metric
from entities.options.embedding_layer_options import EmbeddingLayerOptions
from entities.options.pretrained_representations_options import PretrainedRepresentationsOptions
from entities.batch_representation import BatchRepresentation
from entities.data_output_log import DataOutputLog

from models.model_base import ModelBase
from models.embedding.embedding_layer import EmbeddingLayer

from services.arguments.postocr_characters_arguments_service import PostOCRCharactersArgumentsService
from services.data_service import DataService
from services.metrics_service import MetricsService
from services.vocabulary_service import VocabularyService
from services.file_service import FileService


class CharToCharModel(ModelBase):
    def __init__(
            self,
            arguments_service: PostOCRCharactersArgumentsService,
            vocabulary_service: VocabularyService,
            data_service: DataService,
            metrics_service: MetricsService,
            file_service: FileService):
        super().__init__(data_service, arguments_service)

        self._vocabulary_service = vocabulary_service
        self._metrics_service = metrics_service
        self._arguments_service = arguments_service

        embedding_layer_options = EmbeddingLayerOptions(
            device=arguments_service.device,
            pretrained_representations_options=PretrainedRepresentationsOptions(
                include_pretrained_model=arguments_service.include_pretrained_model,
                pretrained_model_size=arguments_service.pretrained_model_size,
                pretrained_weights=arguments_service.pretrained_weights,
                pretrained_max_length=arguments_service.pretrained_max_length,
                pretrained_model=arguments_service.pretrained_model,
                fine_tune_pretrained=arguments_service.fine_tune_pretrained,
                include_fasttext_model=False),
            learn_character_embeddings=arguments_service.learn_new_embeddings,
            output_embedding_type=EmbeddingType.Character,
            character_embeddings_size=arguments_service.embeddings_size,
            vocabulary_size=vocabulary_service.vocabulary_size(),
            dropout=arguments_service.dropout
        )

        self._embedding_layer = EmbeddingLayer(file_service, embedding_layer_options)

        self._metric_types = arguments_service.metric_types

        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(
            self._embedding_layer.output_size,
            arguments_service.hidden_dimension,
            batch_first=True,
            bidirectional=arguments_service.bidirectional)

        multiplication_factor = 2 if arguments_service.bidirectional else 1
        self._output_layer = nn.Linear(arguments_service.hidden_dimension *
                                       multiplication_factor, vocabulary_service.vocabulary_size())

    @overrides
    def forward(self, input_batch: BatchRepresentation, debug=False, **kwargs):

        embedded = self._embedding_layer.forward(input_batch)

        x_packed = pack_padded_sequence(
            embedded, input_batch.character_lengths, batch_first=True)

        packed_output, _ = self.lstm.forward(x_packed)

        rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        output = self._output_layer.forward(rnn_output)

        if output.shape[1] < input_batch.targets.shape[1]:
            padded_output = torch.zeros((output.shape[0], input_batch.targets.shape[1], output.shape[2])).to(
                self._arguments_service.device)
            padded_output[:, :output.shape[1], :] = output
            output = padded_output
        elif output.shape[1] > input_batch.targets.shape[1]:
            padded_targets = torch.zeros((input_batch.targets.shape[0], output.shape[1]), dtype=torch.int64).to(
                self._arguments_service.device)
            padded_targets[:, :input_batch.targets.shape[1]
                           ] = input_batch.targets
            input_batch.targets = padded_targets

        return output, input_batch.targets

    @overrides
    def calculate_accuracies(self, batch: BatchRepresentation, outputs, output_characters=False) -> Dict[MetricType, float]:
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

        output_log = None
        if MetricType.LevenshteinDistance in self._metric_types:
            predicted_strings = [self._vocabulary_service.ids_to_string(
                x) for x in predicted_characters]
            target_strings = [self._vocabulary_service.ids_to_string(
                x) for x in target_characters]

            levenshtein_distance = np.mean([self._metrics_service.calculate_normalized_levenshtein_distance(
                predicted_strings[i], target_strings[i]) for i in range(len(predicted_strings))])

            metrics[MetricType.LevenshteinDistance] = levenshtein_distance

            if output_characters:
                output_log = DataOutputLog()
                ocr_texts = batch.character_sequences.cpu().detach().tolist()
                for i in range(len(ocr_texts)):
                    input_string = self._vocabulary_service.ids_to_string(
                        ocr_texts[i])

                    output_log.add_new_data(
                        input_data=input_string,
                        output_data=predicted_strings[i],
                        true_data=target_strings[i])

        return metrics, output_log

    @overrides
    def compare_metric(self, best_metric: Metric, new_metric: Metric) -> bool:
        if best_metric.is_new:
            return True

        result = best_metric.get_current_loss() > new_metric.get_current_loss()
        return result
