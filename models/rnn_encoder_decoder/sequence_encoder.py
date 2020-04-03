import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from transformers import BertModel

from services.pretrained_representations_service import PretrainedRepresentationsService


class SequenceEncoder(nn.Module):
    def __init__(
            self,
            pretrained_representations_service: PretrainedRepresentationsService,
            embedding_size: int,
            input_size: int,
            hidden_dimension: int,
            number_of_layers: int,
            dropout: float = 0,
            include_pretrained: bool = False,
            pretrained_hidden_size: int = None,
            learn_embeddings: bool = True,
            bidirectional: bool = False,
            use_own_embeddings: bool = True,
            shared_embeddings = None):
        super().__init__()

        assert learn_embeddings or include_pretrained

        self._pretrained_representations_service = pretrained_representations_service
        self._include_pretrained = include_pretrained
        additional_size = pretrained_hidden_size if self._include_pretrained else 0
        self._learn_embeddings = learn_embeddings
        self._bidirectional = bidirectional
        self._use_own_embeddings = use_own_embeddings
        if not self._use_own_embeddings:
            if shared_embeddings is None:
                raise Exception('Shared embeddings not supplied')
            self._shared_embeddings = shared_embeddings

        lstm_input_size = additional_size
        if learn_embeddings:
            if self._use_own_embeddings:
                self.embedding = nn.Embedding(input_size, embedding_size)

            lstm_input_size += embedding_size

        self.rnn = nn.GRU(lstm_input_size, hidden_dimension,
                          number_of_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_batch, lengths, pretrained_representations, offset_lists, debug: bool = False, **kwargs):
        if self._learn_embeddings:
            if self._use_own_embeddings:
                embedded = self.embedding(input_batch)
            else:
                embedded = self._shared_embeddings(input_batch)

            embedded = self.dropout(embedded)

            if self._include_pretrained:
                embedded = self._pretrained_representations_service.add_pretrained_representations_to_character_embeddings(
                    embedded, pretrained_representations, offset_lists)
        else:
            embedded = pretrained_representations

        x_packed = pack_padded_sequence(embedded, lengths, batch_first=True)

        _, hidden = self.rnn.forward(x_packed)

        if self._bidirectional:
            hidden = torch.cat(
                (hidden[0, :, :], hidden[1, :, :]), dim=1).unsqueeze(0)

        return hidden
