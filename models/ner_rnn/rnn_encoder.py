import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from typing import List, Dict

from models.ner_rnn.rnn_attention import RNNAttention

from services.arguments.ner_arguments_service import NERArgumentsService
from services.tokenizer_service import TokenizerService

from overrides import overrides


class RNNEncoder(nn.Module):
    def __init__(
            self,
            number_of_tags: int,
            use_attention: bool,
            include_pretrained_model: bool,
            pretrained_model_size: int,
            learn_new_embeddings: bool,
            vocabulary_size: int,
            embeddings_size: int,
            dropout: float,
            hidden_dimension: int,
            bidirectional: bool):
        super().__init__()

        self._include_pretrained = include_pretrained_model
        additional_size = pretrained_model_size if self._include_pretrained else 0
        self._learn_embeddings = learn_new_embeddings

        # maps each token to an embedding_dim vector
        rnn_input_size = additional_size
        if self._learn_embeddings:
            self.embedding = nn.Embedding(
                vocabulary_size, embeddings_size)
            self.embedding_dropout = nn.Dropout(dropout)
            rnn_input_size += embeddings_size

        # the LSTM takes embedded sentence
        self.rnn = nn.LSTM(
            rnn_input_size, hidden_dimension, batch_first=True, bidirectional=bidirectional)

        self.rnn_dropout = nn.Dropout(dropout)
        multiplier = 2 if bidirectional else 1

        self._use_attention = use_attention
        if use_attention:
            attention_dimension = hidden_dimension * multiplier
            self.attention = RNNAttention(
                attention_dimension, attention_dimension, attention_dimension)

        self.hidden2tag = nn.Linear(
            hidden_dimension * multiplier, number_of_tags)

        self.bidirectional = bidirectional

    @overrides
    def forward(
            self,
            sequences,
            lengths,
            pretrained_representations,
            debug=False,
            **kwargs):

        if self._learn_embeddings:
            embedded = self.embedding_dropout(self.embedding(sequences))

            if self._include_pretrained:
                embedded = torch.cat(
                    (embedded, pretrained_representations), dim=2)
        else:
            embedded = pretrained_representations

        x_packed = pack_padded_sequence(embedded, lengths, batch_first=True)

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

        output = self.hidden2tag.forward(rnn_output)

        return output
