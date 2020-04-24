import torch
from torch import nn

from typing import List

from overrides import overrides
import fasttext

from entities.batch_representations.base_batch_representation import BaseBatchRepresentation

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService
from services.pretrained_representations_service import PretrainedRepresentationsService
from services.tokenizer_service import TokenizerService


class EmbeddingLayer(nn.Module):
    def __init__(
            self,
            pretrained_representations_service: PretrainedRepresentationsService,
            device: str,
            learn_new_embeddings: bool,
            include_pretrained_model: bool,
            pretrained_model_size: int,
            include_fasttext_model: bool,
            fasttext_model_size: int,
            vocabulary_size: int,
            new_embeddings_size: int,
            dropout: float = 0.0):
        super().__init__()

        self._pretrained_representations_service = pretrained_representations_service

        self._include_pretrained = include_pretrained_model
        self._include_fasttext_model = include_fasttext_model

        self.output_size = 0
        if self._include_pretrained:
            self.output_size += pretrained_model_size

        if self._include_fasttext_model:
            self.output_size += fasttext_model_size

        self._learn_new_embeddings = learn_new_embeddings

        if self._learn_new_embeddings:
            self._embedding = nn.Embedding(
                vocabulary_size,
                new_embeddings_size)
            self._embedding_dropout = nn.Dropout(dropout)
            self.output_size += new_embeddings_size

        self._device = device

    @overrides
    def forward(
            self,
            batch_representation: BaseBatchRepresentation):

        if self._learn_new_embeddings:
            embedded = self._embedding_dropout(
                self._embedding(batch_representation.sequences))

            if self._include_pretrained:
                pretrained_representations = self._pretrained_representations_service.get_pretrained_representation(
                    batch_representation.sequences)
                embedded = torch.cat(
                    (embedded, pretrained_representations), dim=2)

            if self._include_fasttext_model:
                fasttext_tensor = self._pretrained_representations_service.get_fasttext_representation(
                    batch_representation.tokens)

                embedded = torch.cat((embedded, fasttext_tensor), dim=2)
        else:
            embedded = pretrained_representations

        return embedded