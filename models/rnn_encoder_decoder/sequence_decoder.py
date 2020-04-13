import torch
from torch import nn
from torch.functional import F

from overrides import overrides


class SequenceDecoder(nn.Module):
    def __init__(
            self,
            embedding_size: int,
            hidden_dimension: int,
            number_of_layers: int,
            output_dimension: int,
            attention: nn.Module,
            dropout: float = 0,
            use_own_embeddings: bool = True,
            shared_embeddings=None):
        super().__init__()

        self._use_own_embeddings = use_own_embeddings
        if not self._use_own_embeddings:
            if shared_embeddings is None:
                raise Exception('Shared embeddings not supplied')
            self._shared_embeddings = shared_embeddings

        if self._use_own_embeddings:
            self.embedding = nn.Embedding(output_dimension, embedding_size)

        self.rnn = nn.GRU(embedding_size + hidden_dimension,
                          hidden_dimension, number_of_layers, batch_first=True)

        self.attention = attention

        self.fc_out = nn.Linear(
            embedding_size + hidden_dimension * 2, output_dimension)
        self.dropout = nn.Dropout(dropout)

    @overrides
    def forward(self, input_sequence, hidden, encoder_context):
        input_sequence = input_sequence.unsqueeze(1)

        if self._use_own_embeddings:
            embedded = self.embedding(input_sequence)
        else:
            embedded = self._shared_embeddings(input_sequence)

        embedded = self.dropout(embedded)

        attention_result = self.attention(hidden, encoder_context)
        attention_result = attention_result.unsqueeze(1)

        weighted_context = torch.bmm(attention_result, encoder_context)
        rnn_input = torch.cat((embedded, weighted_context), dim=2)
        output, hidden = self.rnn.forward(rnn_input, hidden)

        output = torch.cat(
            (embedded.squeeze(1), hidden.squeeze(0), encoder_context.squeeze(1)), dim=1)

        prediction = self.fc_out(output)
        return prediction, hidden
