import os
from typing import Callable

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from overrides import overrides

from entities.metric import Metric
from entities.batch_representation import BatchRepresentation
from entities.options.embedding_layer_options import EmbeddingLayerOptions

from enums.embedding_type import EmbeddingType

from models.model_base import ModelBase
from models.embedding.embedding_layer import EmbeddingLayer

from services.arguments.semantic_arguments_service import SemanticArgumentsService
from services.vocabulary_service import VocabularyService
from services.data_service import DataService
from services.process.cbow_process_service import CBOWProcessService


class CBOWModel(ModelBase):
    def __init__(
            self,
            arguments_service: SemanticArgumentsService,
            vocabulary_service: VocabularyService,
            data_service: DataService,
            process_service: CBOWProcessService):
        super().__init__(data_service, arguments_service)

        self._arguments_service = arguments_service
        self._mask_token_idx = process_service._mask_idx

        embedding_layer_options = EmbeddingLayerOptions(
            device=arguments_service.device,
            vocabulary_size=vocabulary_service.vocabulary_size(),
            learn_character_embeddings=True,
            character_embeddings_size=arguments_service.word_embeddings_size,
            output_embedding_type=EmbeddingType.Character)

        self._embedding_layer = EmbeddingLayer(embedding_layer_options)

        self._rnn_layer = nn.LSTM(
            input_size=self._embedding_layer.output_size,
            hidden_size=arguments_service.rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False)

        self._output_layer = nn.Linear(
            in_features=arguments_service.rnn_hidden_size,
            out_features=vocabulary_service.vocabulary_size())

    @overrides
    def forward(self, input_batch: BatchRepresentation, **kwargs):
        embeddings = self._embedding_layer.forward(input_batch)

        x_packed = pack_padded_sequence(
            embeddings, input_batch.character_lengths, batch_first=True)

        packed_output, hidden = self._rnn_layer.forward(x_packed)

        rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        output_result = self._output_layer.forward(rnn_output)

        mask_indices = torch.where(input_batch.character_sequences == self._mask_token_idx)[1]
        if mask_indices.shape[0] > 0:
            mask_indices = mask_indices.unsqueeze(-1).repeat(1, output_result.shape[2]).unsqueeze(1)
            gathered_result = output_result.gather(1, mask_indices).squeeze()
        else:
            gathered_result = output_result

        return gathered_result, rnn_output, input_batch.targets

    @overrides
    def compare_metric(self, best_metric: Metric, new_metrics: Metric) -> bool:
        if best_metric.is_new or best_metric.get_current_loss() >= new_metrics.get_current_loss():
            return True

        return False