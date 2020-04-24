import torch
from torch import nn

from typing import List

from overrides import overrides
import fasttext

from entities.batch_representations.base_batch_representation import BaseBatchRepresentation

from enums.embedding_type import EmbeddingType

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService
from services.pretrained_representations_service import PretrainedRepresentationsService
from services.tokenizer_service import TokenizerService


class EmbeddingLayer(nn.Module):
    def __init__(
            self,
            pretrained_representations_service: PretrainedRepresentationsService,
            device: str,
            learn_subword_embeddings: bool = False,
            include_pretrained_model: bool = False,
            pretrained_model_size: int = None,
            include_fasttext_model: bool = False,
            fasttext_model_size: int = None,
            vocabulary_size: int = None,
            subword_embeddings_size: int = None,
            learn_character_embeddings: int = False,
            character_embeddings_size: int = None,
            output_embedding_type: EmbeddingType = EmbeddingType.SubWord,
            dropout: float = 0.0):
        super().__init__()

        self._pretrained_representations_service = pretrained_representations_service
        self._output_embedding_type = output_embedding_type

        self._include_pretrained = include_pretrained_model
        self._include_fasttext_model = include_fasttext_model

        self.output_size = 0
        if self._include_pretrained:
            self.output_size += pretrained_model_size

        if self._include_fasttext_model:
            self.output_size += fasttext_model_size

        self._learn_subword_embeddings = learn_subword_embeddings

        if self._learn_subword_embeddings:
            self._token_embedding = nn.Embedding(
                vocabulary_size,
                subword_embeddings_size)
            self._token_embedding_dropout = nn.Dropout(dropout)
            self.output_size += subword_embeddings_size

        self._learn_character_embeddings = learn_character_embeddings
        if self._learn_character_embeddings:
            self._character_embedding = nn.Embedding(
                vocabulary_size,
                character_embeddings_size)
            self._character_embedding_dropout = nn.Dropout(dropout)
            self.output_size += character_embeddings_size

        self._device = device

    @overrides
    def forward(
            self,
            batch_representation: BaseBatchRepresentation):

        result = None
        if self._learn_subword_embeddings:
            subword_embeddings = self._token_embedding.forward(
                batch_representation.subword_sequences)
            subword_embeddings = self._token_embedding_dropout.forward(
                subword_embeddings)

        if self._learn_character_embeddings:
            character_embeddings = self._character_embedding.forward(
                batch_representation.character_sequences)
            character_embeddings = self._character_embedding_dropout.forward(
                character_embeddings)

        if self._include_pretrained:
            pretrained_embeddings = self._pretrained_representations_service.get_pretrained_representation(
                batch_representation.subword_sequences)

        if self._include_fasttext_model:
            fasttext_embeddings = self._pretrained_representations_service.get_fasttext_representation(
                batch_representation.tokens)

        if self._output_embedding_type == EmbeddingType.Character:
            result = character_embeddings

            if result is None and not self._include_pretrained and not self._include_fasttext_model:
                raise Exception('Invalid configuration')

            if self._learn_subword_embeddings:
                # TODO Concat sub-word embeddings to character embeddings
                pass

            if self._include_pretrained:
                result = self._add_subword_to_character_embeddings(
                    result,
                    pretrained_embeddings,
                    batch_representation.offset_lists)

            if self._include_fasttext_model:
                result = self._add_subword_to_character_embeddings(
                    result,
                    fasttext_embeddings,
                    batch_representation.offset_lists)

        elif self._output_embedding_type == EmbeddingType.SubWord:
            result = subword_embeddings

            if result is None and not self._include_pretrained and not self._include_fasttext_model:
                raise Exception('Invalid configuration')

            if self._learn_character_embeddings:
                pass # TODO Concat character embeddings to sub-word embeddings

            if self._include_pretrained:
                result = torch.cat((result, pretrained_embeddings), dim=2)

            if self._include_fasttext_model:
                result = torch.cat((result, fasttext_embeddings), dim=2)

        return result


    def _add_subword_to_character_embeddings(
            self,
            character_embeddings,
            subword_embeddings,
            offset_lists):
        batch_size = character_embeddings.shape[0]
        pretrained_embedding_size = subword_embeddings.shape[2]

        new_character_embeddings = torch.zeros(
            (batch_size, character_embeddings.shape[1], character_embeddings.shape[2] + subword_embeddings.shape[2])).to(self._arguments_service.device)

        new_character_embeddings[:, :, :character_embeddings.shape[2]] = character_embeddings

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

                    new_character_embeddings[i, k, -pretrained_embedding_size:
                                 ] = subword_embeddings[i, p_i]

                if offset[0] == offset[1]:
                    inserted_count += 1

        return new_character_embeddings
