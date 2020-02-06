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
        self.rnn = nn.GRU(embedding_size + hidden_dimension,
                          hidden_dimension, number_of_layers, batch_first=True)
        self.fc_out = nn.Linear(
            embedding_size + hidden_dimension * 2, output_dimension)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)

        emb_con = torch.cat((embedded, context), dim=2)

        output, hidden = self.rnn.forward(emb_con, hidden)

        output = torch.cat(
            (embedded.squeeze(1), hidden.squeeze(0), context.squeeze(1)), dim=1)

        prediction = self.fc_out(output)
        return prediction, hidden
