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

        batch_size = encoder_context.shape[0]
        src_len = encoder_context.shape[1]

        hidden = hidden.permute(1, 0, 2)

        energy = torch.tanh(
            self._attention.forward(torch.cat((hidden, encoder_context), dim=2)))

        attention = self.v.forward(energy).squeeze(2)

        result = F.softmax(attention, dim=1)
        return result
