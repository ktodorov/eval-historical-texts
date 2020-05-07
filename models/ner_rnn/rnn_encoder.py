import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from typing import List, Dict

from enums.entity_tag_type import EntityTagType

from entities.batch_representation import BatchRepresentation
from entities.options.rnn_encoder_options import RNNEncoderOptions
from entities.options.embedding_layer_options import EmbeddingLayerOptions

from models.ner_rnn.rnn_attention import RNNAttention
from models.embedding.embedding_layer import EmbeddingLayer

from services.arguments.ner_arguments_service import NERArgumentsService
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.file_service import FileService

from overrides import overrides
from models.model_base import ModelBase


class RNNEncoder(ModelBase):
    def __init__(
            self,
            file_service: FileService,
            rnn_encoder_options: RNNEncoderOptions):
        super().__init__()

        self.device = rnn_encoder_options.device

        embedding_layer_options = EmbeddingLayerOptions(
            device=rnn_encoder_options.device,
            pretrained_representations_options=rnn_encoder_options.pretrained_representations_options,
            learn_subword_embeddings=rnn_encoder_options.learn_new_embeddings,
            subword_embeddings_size=rnn_encoder_options.embeddings_size,
            vocabulary_size=rnn_encoder_options.vocabulary_size,
            merge_subword_embeddings=rnn_encoder_options.merge_subword_embeddings,
            learn_character_embeddings=rnn_encoder_options.learn_character_embeddings,
            character_embeddings_size=rnn_encoder_options.character_embeddings_size,
            character_rnn_hidden_size=rnn_encoder_options.character_hidden_size,
            dropout=rnn_encoder_options.dropout,
            learn_manual_features=rnn_encoder_options.learn_manual_features,
            manual_features_count=rnn_encoder_options.manual_features_count)

        self._embedding_layer = EmbeddingLayer(
            file_service,
            embedding_layer_options)

        # the LSTM takes embedded sentence
        self.rnn = nn.LSTM(
            self._embedding_layer.output_size,
            rnn_encoder_options.hidden_dimension,
            num_layers=rnn_encoder_options.number_of_layers,
            batch_first=True,
            bidirectional=rnn_encoder_options.bidirectional)

        self.rnn_dropout = nn.Dropout(rnn_encoder_options.dropout)
        multiplier = 2 if rnn_encoder_options.bidirectional else 1

        self._use_attention = rnn_encoder_options.use_attention
        if rnn_encoder_options.use_attention:
            attention_dimension = rnn_encoder_options.hidden_dimension * multiplier
            self.attention = RNNAttention(
                attention_dimension, attention_dimension, attention_dimension)

        self._output_layers = nn.ModuleList([
            nn.Linear(rnn_encoder_options.hidden_dimension *
                      multiplier, number_of_tags)
            for number_of_tags in rnn_encoder_options.number_of_tags.values()
        ])

        self._entity_tag_types: List[EntityTagType] = list(
            rnn_encoder_options.number_of_tags.keys())

        self.bidirectional = rnn_encoder_options.bidirectional

    @overrides
    def forward(
            self,
            batch_representation: BatchRepresentation,
            debug=False,
            **kwargs):

        embedded = self._embedding_layer.forward(batch_representation)

        x_packed = pack_padded_sequence(
            embedded, batch_representation.subword_lengths, batch_first=True)

        packed_output, hidden = self.rnn.forward(x_packed)

        rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        rnn_output = self.rnn_dropout.forward(rnn_output)

        if self._use_attention:
            if isinstance(hidden, tuple):  # LSTM
                hidden = hidden[1]  # take the cell state

            if self.bidirectional:  # need to concat the last 2 hidden layers
                hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
            else:
                hidden = hidden[-1]

            energy, linear_combination = self.attention.forward(
                hidden,
                rnn_output,
                rnn_output)

            linear_combination = linear_combination.expand_as(rnn_output)
            rnn_output = linear_combination * rnn_output

        outputs: Dict[EntityTagType, torch.Tensor] = {}
        for i, entity_tag_type in enumerate(self._entity_tag_types):
            output = self._output_layers[i].forward(rnn_output)
            outputs[entity_tag_type] = output

        return outputs, batch_representation.subword_lengths
