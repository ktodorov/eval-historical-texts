import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import sentencepiece as spm
from sentencepiece import SentencePieceProcessor


class MultiFitEncoder(nn.Module):
    def __init__(
            self,
            embedding_size: int,
            input_size: int,
            hidden_dimension: int,
            number_of_layers: int,
            dropout: float = 0):
        super(MultiFitEncoder, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_dimension, number_of_layers,
                           dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_batch, lengths, **kwargs):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(input_batch))
        # embedded = [src len, batch size, emb dim]

        x_packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        outputs, (hidden, cell) = self.rnn(x_packed)
        # outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device='cuda')
