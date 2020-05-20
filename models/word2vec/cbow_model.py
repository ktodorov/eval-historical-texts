import os
from typing import Callable

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from overrides import overrides

from entities.metric import Metric
from entities.batch_representation import BatchRepresentation
from entities.options.embedding_layer_options import EmbeddingLayerOptions
from entities.options.pretrained_representations_options import PretrainedRepresentationsOptions

from enums.embedding_type import EmbeddingType

from models.model_base import ModelBase
from models.embedding.embedding_layer import EmbeddingLayer

from services.arguments.semantic_arguments_service import SemanticArgumentsService
from services.vocabulary_service import VocabularyService
from services.data_service import DataService
from services.file_service import FileService
from services.process.cbow_process_service import CBOWProcessService


class CBOWModel(ModelBase):
    def __init__(
            self,
            arguments_service: SemanticArgumentsService,
            vocabulary_service: VocabularyService,
            data_service: DataService,
            process_service: CBOWProcessService,
            file_service: FileService,
            use_only_embeddings: bool = False):
        super().__init__(data_service, arguments_service)

        self._arguments_service = arguments_service
        self._vocabulary_service = vocabulary_service
        self._mask_token_idx = process_service._mask_idx

        pretrained_word_weights = None
        if not arguments_service.evaluate and not arguments_service.resume_training:
            pretrained_word_weights = process_service.get_pretrained_embedding_weights()

        pretrained_weight_matrix_dim = 300
        if pretrained_word_weights is not None:
            pretrained_weight_matrix_dim = pretrained_word_weights.shape[1]

        embedding_layer_options = EmbeddingLayerOptions(
            device=arguments_service.device,
            pretrained_representations_options=PretrainedRepresentationsOptions(
                include_pretrained_model=False),
            vocabulary_size=vocabulary_service.vocabulary_size(),
            learn_word_embeddings=True,
            word_embeddings_size=pretrained_weight_matrix_dim,
            pretrained_word_weights=pretrained_word_weights,
            output_embedding_type=EmbeddingType.Word)

        self._embedding_layer = EmbeddingLayer(
            file_service, embedding_layer_options)

        self._mapping_layer = nn.Linear(
            in_features=6 * pretrained_weight_matrix_dim,
            out_features=128)

        self._output_layer = nn.Linear(
            in_features=128,
            out_features=vocabulary_service.vocabulary_size())

        self._use_only_embeddings = use_only_embeddings

    @overrides
    def forward(self, input_batch: BatchRepresentation, **kwargs):
        embeddings = self._embedding_layer.forward(input_batch)

        if self._use_only_embeddings:
            return embeddings

        embeddings = embeddings.view(input_batch.batch_size, -1)
        mapped_embeddings = self._mapping_layer.forward(embeddings)
        output_result = self._output_layer.forward(mapped_embeddings)

        return output_result, input_batch.targets

    @overrides
    def compare_metric(self, best_metric: Metric, new_metrics: Metric) -> bool:
        if best_metric.is_new or best_metric.get_current_loss() >= new_metrics.get_current_loss():
            return True

        return False