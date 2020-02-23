import torch
from torch import nn

from models.multi_head_attention_layer import MultiHeadAttentionLayer
from models.positionwise_feedforward_layer import PositionwiseFeedforwardLayer


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 include_pretrained: bool = False,
                 pretrained_hidden_size: int = None):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim,
            n_heads,
            dropout,
            device,
            include_pretrained=include_pretrained,
            pretrained_hidden_size=pretrained_hidden_size)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(
            trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, attention
