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
            attention,
            dropout: float = 0):
        super(MultiFitDecoder, self).__init__()

        self.embedding = nn.Embedding(output_dimension, embedding_size)
        self.attention = attention

        self.rnn = nn.GRU((hidden_dimension * 2) + embedding_size, hidden_dimension,
                          number_of_layers, batch_first=True)
        self.fc_out = nn.Linear(
            (hidden_dimension * 2) + hidden_dimension + embedding_size, output_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))

        a = self.attention(hidden, encoder_outputs)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        # encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        # weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        #rnn_input = [batch size, 1, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn.forward(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden.permute(1, 0, 2)).all()

        # prediction = self.fc_out(output.squeeze(1))

        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)

        prediction = self.fc_out(
            torch.cat((output, weighted, embedded), dim=1))

        return prediction, hidden.squeeze(0)
