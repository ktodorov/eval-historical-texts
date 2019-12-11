import torch
from torch import nn

import sentencepiece as spm
from sentencepiece import SentencePieceProcessor


class MultiFitEncoder(nn.Module):
    def __init__(self):
        super(MultiFitEncoder, self).__init__()

        emb_dim = 256
        input_size = 1000
        hidden_dim = 256
        dropout = 0

        self.embedding = nn.Embedding(input_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, 1,
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_batch, **kwargs):
        # src = [src len, batch size]
        src = input_batch
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device='cuda')
