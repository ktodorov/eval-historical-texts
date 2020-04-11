import torch
from torch import nn
from torch.functional import F

class SequenceAttention(nn.Module):
    def __init__(
            self,
            encoder_hidden_dimension: int,
            decoder_hidden_dimension: int):
        super().__init__()

        self._attention = nn.Linear(
            (encoder_hidden_dimension * 2) + (decoder_hidden_dimension * 2),
            decoder_hidden_dimension)

        self.v = nn.Linear(
            decoder_hidden_dimension,
            1,
            bias=False)

    def forward(self, hidden, encoder_context):

        # hidden = [batch size, dec hid dim]
        # encoder_context = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_context.shape[0]
        src_len = encoder_context.shape[1]

        # repeat decoder hidden state src_len times
        # hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # encoder_context = encoder_context.permute(1, 0, 2)
        hidden = hidden.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_context = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(
            self._attention.forward(torch.cat((hidden, encoder_context), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v.forward(energy).squeeze(2)

        # attention= [batch size, src len]

        result = F.softmax(attention, dim=1)
        return result
