import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from transformers import BertModel


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

        self._include_bert = include_bert
        additional_size = 768 if self._include_bert else 0
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size + additional_size, hidden_dimension, number_of_layers,
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        if self._include_bert and pretrained_weights:
            self._pretrained_model = BertModel.from_pretrained(
                pretrained_weights)
            self._pretrained_model.eval()

    def forward(self, input_batch, lengths, **kwargs):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(input_batch))
        # embedded = [src len, batch size, emb dim]

        if self._include_bert:
            with torch.no_grad():
                pretrained_outputs = self._pretrained_model.forward(
                    input_batch)  # , lm_labels=input_batch)
                embedded = torch.cat((embedded, pretrained_outputs[0]), dim=2)

        x_packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        outputs, (hidden, cell) = self.rnn(x_packed)
        # outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        hidden = torch.cat(
            (hidden[0, :, :], hidden[1, :, :]), dim=1).unsqueeze(0)
        cell = torch.cat((cell[0, :, :], cell[1, :, :]), dim=1).unsqueeze(0)

        return hidden, cell
