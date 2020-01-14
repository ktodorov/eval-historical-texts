import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import sentencepiece as spm
from sentencepiece import SentencePieceProcessor

from transformers import XLNetModel

class MultiFitEncoder(nn.Module):
    def __init__(
            self,
            embedding_size: int,
            input_size: int,
            hidden_dimension: int,
            number_of_layers: int,
            dropout: float = 0,
            include_bert: bool = False,
            pretrained_weights: str = None):
        super(MultiFitEncoder, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_dimension, number_of_layers,
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self._include_bert = include_bert
        if self._include_bert and pretrained_weights:
            self._xlnet_model = XLNetModel.from_pretrained(pretrained_weights)
            self._xlnet_model.eval()

    def forward(self, input_batch, lengths, masked_inputs, masked_labels, **kwargs):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(input_batch))
        # embedded = [src len, batch size, emb dim]

        x_packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        outputs, (hidden, cell) = self.rnn(x_packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        xlnet_hidden = None
        if self._include_bert:
            with torch.no_grad():
                xlnet_outputs = self._xlnet_model.forward(masked_inputs, attention_mask=masked_labels)
                xlnet_hidden = xlnet_outputs[0]

        return hidden, cell, xlnet_hidden