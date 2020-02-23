import torch
from torch import nn

from models.transformer_encoder_layer import TransformerEncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=2000,
                 include_pretrained: bool = False,
                 pretrained_hidden_size: int = None):
        super().__init__()

        self.device = device
        self.include_pretrained = include_pretrained

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        additional_size = pretrained_hidden_size if include_pretrained else 0

        self.layers = nn.ModuleList([TransformerEncoderLayer(hid_dim + additional_size,
                                                             n_heads,
                                                             pf_dim,
                                                             dropout,
                                                             device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask, pretrained_representations=None):

        # src = [batch size, src len]
        # src_mask = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(
            0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, src len]

        src = self.dropout(
            (self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]

        if self.include_pretrained:
            src = torch.cat((src, pretrained_representations), dim=2)

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]

        return src
