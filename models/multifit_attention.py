import torch
from torch import nn
from torch.functional import F

class MultiFitAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.attn = nn.Linear((hidden_size * 2) + (hidden_size), hidden_size)
        self.v = nn.Parameter(torch.randn(hidden_size))

    def forward(self, hidden, encoder_outputs):

        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # encoder_outputs = encoder_outputs.permute(0, 1, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        combined = torch.cat((hidden, encoder_outputs), dim=2)
        attn_output = self.attn(combined)
        energy = torch.tanh(attn_output)

        # energy = [batch size, src len, dec hid dim]

        energy = energy.permute(0, 2, 1)

        # energy = [batch size, dec hid dim, src len]

        # v = [dec hid dim]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # v = [batch size, 1, dec hid dim]

        attention = torch.bmm(v, energy).squeeze(1)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)
