import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from typing import List, Dict

from models.ner_rnn.rnn_attention import RNNAttention
from models.embedding.embedding_layer import EmbeddingLayer

from services.arguments.ner_arguments_service import NERArgumentsService
from services.tokenizer_service import TokenizerService
from services.pretrained_representations_service import PretrainedRepresentationsService

from overrides import overrides


class RNNEncoder(nn.Module):
    def __init__(
            self,
            pretrained_representations_service: PretrainedRepresentationsService,
            device: str,
            number_of_tags: int,
            use_attention: bool,
            include_pretrained_model: bool,
            pretrained_model_size: int,
            include_fasttext_model: bool,
            fasttext_model_size: int,
            learn_new_embeddings: bool,
            vocabulary_size: int,
            embeddings_size: int,
            dropout: float,
            hidden_dimension: int,
            bidirectional: bool,
            number_of_layers: int):
        super().__init__()

        self._include_pretrained = include_pretrained_model
        additional_size = pretrained_model_size if self._include_pretrained else 0
        self._learn_embeddings = learn_new_embeddings

        # maps each token to an embedding_dim vector
        rnn_input_size = additional_size

        self.device = device

        self._embedding_layer = EmbeddingLayer(
            pretrained_representations_service=pretrained_representations_service,
            device=device,
            learn_new_embeddings=learn_new_embeddings,
            include_pretrained_model=include_pretrained_model,
            pretrained_model_size=pretrained_model_size,
            include_fasttext_model=include_fasttext_model,
            fasttext_model_size=fasttext_model_size,
            vocabulary_size=vocabulary_size,
            new_embeddings_size=embeddings_size,
            dropout=dropout)

        # the LSTM takes embedded sentence
        self.rnn = nn.LSTM(
            self._embedding_layer.output_size,
            hidden_dimension,
            num_layers=number_of_layers,
            batch_first=True,
            bidirectional=bidirectional)

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
            sequences_strings,
            lengths,
            position_changes,
            targets,
            debug=False,
            **kwargs):

        embedded = self._embedding_layer.forward(sequences, sequences_strings)

        new_embedded, new_lengths, new_targets = self._restore_position_changes(position_changes, embedded, lengths, targets)

        x_packed = pack_padded_sequence(new_embedded, new_lengths, batch_first=True)

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
        return output, new_lengths, new_targets

    def _sort_batch(self, embeddings, lengths, targets):
        lengths, perm_idx = lengths.sort(descending=True)
        embeddings = embeddings[perm_idx]
        targets = targets[perm_idx]

        return embeddings, lengths, targets

    def _restore_position_changes(
            self,
            position_changes,
            embeddings,
            lengths,
            targets):
        batch_size, sequence_length, embeddings_size = embeddings.shape

        new_max_sequence_length = max([len(x.keys()) for x in position_changes])

        new_embeddings = torch.zeros(
            (batch_size, new_max_sequence_length, embeddings_size), dtype=embeddings.dtype).to(self.device)
        new_targets = torch.zeros((batch_size, new_max_sequence_length), dtype=targets.dtype).to(self.device)
        new_lengths = torch.zeros((batch_size), dtype=lengths.dtype).to(self.device)

        for i, current_position_changes in enumerate(position_changes):
            new_lengths[i] = len(current_position_changes.keys())

            for old_position, new_positions in current_position_changes.items():
                if len(new_positions) == 1:
                    new_embeddings[i,old_position,:] = embeddings[i, new_positions[0], :]
                else:
                    new_embeddings[i,old_position,:] = torch.mean(embeddings[i, new_positions], dim=0)

                new_targets[i,old_position] = targets[i, new_positions[0]]

        return new_embeddings, new_lengths, new_targets
