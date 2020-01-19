import torch
from torch import nn
from torch.functional import F


class MultiFitDecoder(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        hidden_dimension: int,
        number_of_layers: int,
        output_dimension: int,
        dropout: float = 0):
        super(MultiFitDecoder, self).__init__()

        self.embedding = nn.Embedding(output_dimension, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_dimension, number_of_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dimension, output_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn.forward(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell
