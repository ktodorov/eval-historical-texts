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
            include_pretrained: bool = False,
            pretrained_hidden_size: int = None,
            pretrained_weights: str = None):
        super(MultiFitEncoder, self).__init__()

        self._include_pretrained = include_pretrained
        additional_size = pretrained_hidden_size if self._include_pretrained else 0
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size + additional_size, hidden_dimension, number_of_layers,
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_batch, lengths, pretrained_representations, debug: bool = False, **kwargs):
        embedded = self.dropout(self.embedding(input_batch))
        self._print_debug_embedded_statistics(debug, embedded)

        if self._include_pretrained:
            embedded = torch.cat((embedded, pretrained_representations), dim=2)

        x_packed = pack_padded_sequence(embedded, lengths, batch_first=True)

        outputs, (hidden, cell) = self.rnn.forward(x_packed)
        self._print_debug_hidden_statistics(debug, hidden, cell)

        hidden = torch.cat(
            (hidden[0, :, :], hidden[1, :, :]), dim=1).unsqueeze(0)
        cell = torch.cat((cell[0, :, :], cell[1, :, :]), dim=1).unsqueeze(0)

        return hidden, cell

    def _print_debug_embedded_statistics(self, debug: bool, embedded):
        if debug:
            print(f'embedded nans[0]: {torch.isnan(embedded).sum(dim=0)}')
            print(f'embedded nans[1]: {torch.isnan(embedded).sum(dim=1)}')
            print(f'embedded nans[2]: {torch.isnan(embedded).sum(dim=2)}')

    def _print_debug_hidden_statistics(self, debug: bool, hidden, cell):
        if debug:
            print(f'hidden nans: {torch.isnan(hidden).sum().item()}')
            print(f'cell nans: {torch.isnan(cell).sum().item()}')
