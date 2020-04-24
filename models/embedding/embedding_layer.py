import torch
from torch import nn

from typing import List

from overrides import overrides
import fasttext

from entities.batch_representation import BatchRepresentation

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
            merge_subword_embeddings: bool = False,
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
        self._merge_subword_embeddings = merge_subword_embeddings

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
            batch_representation: BatchRepresentation):

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

        result_embeddings = None
        if self._output_embedding_type == EmbeddingType.Character:
            result_embeddings = character_embeddings

            if result_embeddings is None and not self._include_pretrained and not self._include_fasttext_model:
                raise Exception('Invalid configuration')

            if self._learn_subword_embeddings:
                # TODO Concat sub-word embeddings to character embeddings
                pass

            if self._include_pretrained:
                result_embeddings = self._add_subword_to_character_embeddings(
                    result_embeddings,
                    pretrained_embeddings,
                    batch_representation.offset_lists)

            if self._include_fasttext_model:
                result_embeddings = self._add_subword_to_character_embeddings(
                    result_embeddings,
                    fasttext_embeddings,
                    batch_representation.offset_lists)

        elif self._output_embedding_type == EmbeddingType.SubWord:
            result_embeddings = subword_embeddings

            if result_embeddings is None and not self._include_pretrained and not self._include_fasttext_model:
                raise Exception('Invalid configuration')

            if self._learn_character_embeddings:
                result_embeddings = self._add_character_to_subword_embeddings(
                    batch_representation.batch_size,
                    result_embeddings,
                    character_embeddings,
                    batch_representation.subword_characters_count)

            if self._include_pretrained:
                result_embeddings = torch.cat(
                    (result_embeddings, pretrained_embeddings), dim=2)

            if self._include_fasttext_model:
                result_embeddings = torch.cat(
                    (result_embeddings, fasttext_embeddings), dim=2)

            if self._merge_subword_embeddings and batch_representation.position_changes is not None:
                (result_embeddings,
                 batch_representation._subword_lengths,
                 batch_representation._targets) = self._restore_position_changes(
                    position_changes=batch_representation.position_changes,
                    embeddings=result_embeddings,
                    lengths=batch_representation.subword_lengths,
                    targets=batch_representation.targets)

        return result_embeddings

    def _add_subword_to_character_embeddings(
            self,
            character_embeddings,
            subword_embeddings,
            offset_lists):
        batch_size = character_embeddings.shape[0]
        pretrained_embedding_size = subword_embeddings.shape[2]

        new_character_embeddings = torch.zeros(
            (batch_size, character_embeddings.shape[1], character_embeddings.shape[2] + subword_embeddings.shape[2])).to(self._arguments_service.device)

        new_character_embeddings[:, :,
                                 :character_embeddings.shape[2]] = character_embeddings

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

    def _restore_position_changes(
            self,
            position_changes,
            embeddings,
            lengths,
            targets):
        batch_size, sequence_length, embeddings_size = embeddings.shape

        new_max_sequence_length = max(
            [len(x.keys()) for x in position_changes])

        new_embeddings = torch.zeros(
            (batch_size, new_max_sequence_length, embeddings_size), dtype=embeddings.dtype).to(self._device)
        new_targets = torch.zeros(
            (batch_size, new_max_sequence_length), dtype=targets.dtype).to(self._device)
        new_lengths = torch.zeros(
            (batch_size), dtype=lengths.dtype).to(self._device)

        for i, current_position_changes in enumerate(position_changes):
            new_lengths[i] = len(current_position_changes.keys())

            for old_position, new_positions in current_position_changes.items():
                if len(new_positions) == 1:
                    new_embeddings[i, old_position,
                                   :] = embeddings[i, new_positions[0], :]
                else:
                    new_embeddings[i, old_position, :] = torch.mean(
                        embeddings[i, new_positions], dim=0)

                new_targets[i, old_position] = targets[i, new_positions[0]]

        return new_embeddings, new_lengths, new_targets

    def _add_character_to_subword_embeddings(
            self,
            batch_size: int,
            subword_embeddings: torch.Tensor,
            character_embeddings: torch.Tensor,
            subword_characters_count: List[List[int]]):

        subword_dimension = subword_embeddings.shape[2]
        concat_dimension = subword_dimension + character_embeddings.shape[2]
        result_embeddings = torch.zeros(
            (batch_size, subword_embeddings.shape[1], concat_dimension), device=self._device)
        result_embeddings[:, :, :subword_dimension] = subword_embeddings

        for b in range(batch_size):
            counter = 0
            for index, characters_count in enumerate(subword_characters_count[b]):
                character_indices = [
                    counter + i for i in range(characters_count)]
                if len(character_indices) == 1:
                    result_embeddings[b, :,
                                      subword_dimension:] = character_embeddings[b, character_indices[0], :]
                else:
                    result_embeddings[b, :, subword_dimension:] = torch.mean(character_embeddings[b, character_indices, :])

                counter += characters_count

        return result_embeddings