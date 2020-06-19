import torch
from torch import nn

from typing import List

from overrides import overrides
import fasttext

from entities.batch_representation import BatchRepresentation
from entities.options.embedding_layer_options import EmbeddingLayerOptions

from enums.embedding_type import EmbeddingType

from models.embedding.character_rnn import CharacterRNN
from models.pretrained.pretrained_representations_layer import PretrainedRepresentationsLayer
from models.model_base import ModelBase

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.file_service import FileService


class EmbeddingLayer(ModelBase):
    def __init__(
            self,
            file_service: FileService,
            embedding_layer_options: EmbeddingLayerOptions):
        super().__init__()

        self._output_embedding_type = embedding_layer_options.output_embedding_type
        self._merge_subword_embeddings = embedding_layer_options.merge_subword_embeddings

        self._include_pretrained = embedding_layer_options.pretrained_representations_options.include_pretrained_model
        self._include_fasttext_model = embedding_layer_options.pretrained_representations_options.include_fasttext_model

        self._output_size = 0
        if self._include_pretrained or self._include_fasttext_model:
            self._pretrained_layer = PretrainedRepresentationsLayer(
                file_service=file_service,
                device=embedding_layer_options.device,
                pretrained_representations_options=embedding_layer_options.pretrained_representations_options)

            if self._include_pretrained:
                self._output_size += embedding_layer_options.pretrained_representations_options.pretrained_model_size

            if self._include_fasttext_model:
                self._output_size += embedding_layer_options.pretrained_representations_options.fasttext_model_size

        self._learn_subword_embeddings = embedding_layer_options.learn_subword_embeddings

        if self._learn_subword_embeddings:
            self._subword_embedding = nn.Embedding(
                embedding_layer_options.vocabulary_size,
                embedding_layer_options.subword_embeddings_size)
            self._subword_embedding_dropout = nn.Dropout(
                embedding_layer_options.dropout)
            self._output_size += embedding_layer_options.subword_embeddings_size

        self._learn_character_embeddings = embedding_layer_options.learn_character_embeddings
        if self._learn_character_embeddings:
            if embedding_layer_options.output_embedding_type == EmbeddingType.Character:
                self._character_embedding = nn.Embedding(
                    embedding_layer_options.vocabulary_size,
                    embedding_layer_options.character_embeddings_size)
                self._character_embedding_dropout = nn.Dropout(
                    embedding_layer_options.dropout)
                self._output_size += embedding_layer_options.character_embeddings_size
            else:
                self._character_embedding = CharacterRNN(
                    vocabulary_size=embedding_layer_options.vocabulary_size,
                    character_embedding_size=embedding_layer_options.character_embeddings_size,
                    hidden_size=embedding_layer_options.character_rnn_hidden_size,
                    number_of_layers=1,
                    bidirectional_rnn=True,
                    dropout=0)

                self._output_size += (
                    embedding_layer_options.character_rnn_hidden_size * 2)

        self._learn_word_embeddings = embedding_layer_options.learn_word_embeddings
        if self._learn_word_embeddings:
            if embedding_layer_options.pretrained_word_weights is None:
                self._word_embedding = nn.Embedding(
                    embedding_layer_options.vocabulary_size,
                    embedding_layer_options.word_embeddings_size)
            else:
                self._word_embedding: nn.Embedding = nn.Embedding.from_pretrained(
                    embedding_layer_options.pretrained_word_weights,
                    freeze=False)

            self._word_embedding_dropout = nn.Dropout(
                embedding_layer_options.dropout)
            self._output_size += self._word_embedding.embedding_dim

        self._learn_manual_features = embedding_layer_options.learn_manual_features
        if self._learn_manual_features:
            self._manual_features_layer = nn.Embedding(
                num_embeddings=(
                    embedding_layer_options.manual_features_count * 2) + 1,
                embedding_dim=1)

            self._output_size += embedding_layer_options.manual_features_count

        self._device = embedding_layer_options.device

    @overrides
    def forward(
            self,
            batch_representation: BatchRepresentation, 
            skip_pretrained_representation: bool = False):
        subword_embeddings = None
        word_embeddings = None
        character_embeddings = None

        include_pretrained = self._include_pretrained and batch_representation.subword_sequences is not None

        if self._learn_word_embeddings:
            word_embeddings = self._word_embedding.forward(
                batch_representation.word_sequences)
            word_embeddings = self._word_embedding_dropout.forward(
                word_embeddings)

        if self._learn_subword_embeddings:
            subword_embeddings = self._subword_embedding.forward(
                batch_representation.subword_sequences)
            subword_embeddings = self._subword_embedding_dropout.forward(
                subword_embeddings)

        if self._learn_character_embeddings:
            if self._output_embedding_type == EmbeddingType.Character:
                character_embeddings = self._character_embedding.forward(
                    batch_representation.character_sequences)

                if self._character_embedding_dropout is not None:
                    character_embeddings = self._character_embedding_dropout.forward(
                        character_embeddings)
            else:
                character_embeddings = self._character_embedding.forward(
                    batch_representation.character_sequences,
                    batch_representation.subword_characters_count)

        if include_pretrained and not skip_pretrained_representation:
            pretrained_embeddings = self._pretrained_layer.get_pretrained_representation(
                batch_representation.subword_sequences)

        if self._include_fasttext_model and not skip_pretrained_representation:
            fasttext_embeddings = self._pretrained_layer.get_fasttext_representation(
                batch_representation.tokens)

        if self._learn_manual_features:
            manual_feature_embeddings = self._manual_features_layer.forward(
                batch_representation.manual_features)

            manual_feature_embeddings = manual_feature_embeddings.squeeze(-1)

        result_embeddings = None
        if self._output_embedding_type == EmbeddingType.Character:
            result_embeddings = character_embeddings

            if result_embeddings is None and not include_pretrained and not self._include_fasttext_model:
                raise Exception('Invalid configuration')

            if self._learn_subword_embeddings:
                # TODO Concat sub-word embeddings to character embeddings
                pass

            if include_pretrained and not skip_pretrained_representation:
                result_embeddings = self._add_subword_to_character_embeddings(
                    result_embeddings,
                    pretrained_embeddings,
                    batch_representation.offset_lists)

            if self._include_fasttext_model and not skip_pretrained_representation:
                result_embeddings = self._add_subword_to_character_embeddings(
                    result_embeddings,
                    fasttext_embeddings,
                    batch_representation.offset_lists)

        elif self._output_embedding_type == EmbeddingType.SubWord:
            result_embeddings = None
            if subword_embeddings is not None:
                result_embeddings = subword_embeddings

            if result_embeddings is None and not include_pretrained and not self._include_fasttext_model:
                raise Exception('Invalid configuration')

            if self._learn_character_embeddings:
                result_embeddings = self._add_character_to_subword_embeddings(
                    batch_representation.batch_size,
                    result_embeddings,
                    character_embeddings,
                    batch_representation.subword_characters_count)

            if include_pretrained and not skip_pretrained_representation:
                if result_embeddings is None:
                    result_embeddings = pretrained_embeddings
                else:
                    result_embeddings = torch.cat(
                        (result_embeddings, pretrained_embeddings), dim=2)

            if self._include_fasttext_model:
                if result_embeddings is None:
                    result_embeddings = fasttext_embeddings
                else:
                    result_embeddings = torch.cat(
                        (result_embeddings, fasttext_embeddings), dim=2)

            if self._learn_manual_features:
                if result_embeddings is None:
                    result_embeddings = manual_feature_embeddings
                else:
                    result_embeddings = torch.cat(
                        (result_embeddings, manual_feature_embeddings), dim=2)

            if self._merge_subword_embeddings and batch_representation.position_changes is not None:
                result_embeddings, batch_representation._subword_lengths = self._restore_position_changes(
                    position_changes=batch_representation.position_changes,
                    embeddings=result_embeddings,
                    lengths=batch_representation.subword_lengths)

        elif self._output_embedding_type == EmbeddingType.Word:
            result_embeddings = word_embeddings

        return result_embeddings

    def _add_subword_to_character_embeddings(
            self,
            character_embeddings,
            subword_embeddings,
            offset_lists):
        batch_size = character_embeddings.shape[0]
        pretrained_embedding_size = subword_embeddings.shape[2]

        new_character_embeddings = torch.zeros(
            (batch_size, character_embeddings.shape[1], character_embeddings.shape[2] + subword_embeddings.shape[2])).to(self._device)

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
            lengths):
        batch_size, sequence_length, embeddings_size = embeddings.shape

        new_max_sequence_length = max(
            [len(x.keys()) for x in position_changes])

        new_embeddings = torch.zeros(
            (batch_size, new_max_sequence_length, embeddings_size), dtype=embeddings.dtype).to(self._device)
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

        return new_embeddings, new_lengths

    def _add_character_to_subword_embeddings(
            self,
            batch_size: int,
            subword_embeddings: torch.Tensor,
            character_embeddings: torch.Tensor,
            subword_characters_count: List[List[int]]):

        if subword_embeddings is None:
            return character_embeddings

        subword_dimension = subword_embeddings.shape[2]
        concat_dimension = subword_dimension + character_embeddings.shape[2]

        result_embeddings = torch.cat(
            [subword_embeddings, character_embeddings], dim=-1)

        return result_embeddings

    @property
    def output_size(self) -> int:
        return self._output_size
