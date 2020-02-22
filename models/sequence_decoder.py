import torch
from torch import nn
from torch.functional import F


class SequenceDecoder(nn.Module):
    def __init__(
            self,
            embedding_size: int,
            hidden_dimension: int,
            number_of_layers: int,
            output_dimension: int,
            attention,
            dropout: float = 0):
        super(SequenceDecoder, self).__init__()

        self.embedding = nn.Embedding(output_dimension, embedding_size)

        self.rnn = nn.GRU(embedding_size + (hidden_dimension * 2),
                          hidden_dimension, number_of_layers, batch_first=True)

        self.fc_out = nn.Linear((hidden_dimension * 2) +
                                embedding_size + hidden_dimension, output_dimension)

        self.dropout = nn.Dropout(dropout)
        self.attention = attention

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))

        a = self.attention(hidden, encoder_outputs)

        a = a.unsqueeze(1)

        # encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        # weighted = weighted.permute(1, 0, 2)

        emb_con = torch.cat((embedded, weighted), dim=2)

        output, hidden = self.rnn.forward(emb_con, hidden.unsqueeze(0))

        # assert (output == hidden).all()

        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))

        # output = torch.cat(
        #     (embedded.squeeze(1), hidden.squeeze(0), encoder_outputs.squeeze(1)), dim=1)

        # prediction = self.fc_out(output)
        return prediction, hidden.squeeze(0)
