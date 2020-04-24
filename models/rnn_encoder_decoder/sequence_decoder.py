import torch
from torch import nn
from torch.functional import F

from overrides import overrides

from enums.embedding_type import EmbeddingType
from entities.batch_representations.simple_batch_representation import SimpleBatchRepresentation

from models.embedding.embedding_layer import EmbeddingLayer

class SequenceDecoder(nn.Module):
    def __init__(
            self,
            device: str,
            embedding_size: int,
            hidden_dimension: int,
            number_of_layers: int,
            output_dimension: int,
            attention: nn.Module,
            dropout: float = 0,
            use_own_embeddings: bool = True,
            shared_embedding_layer: EmbeddingLayer = None):
        super().__init__()

        self._device = device

        if not use_own_embeddings:
            if shared_embedding_layer is None:
                raise Exception('Shared embeddings not supplied')
            self._embedding_layer = shared_embedding_layer
        else:
            self._embedding_layer = EmbeddingLayer(
                pretrained_representations_service=None,
                device=device,
                learn_character_embeddings=True,
                include_pretrained_model=False,
                vocabulary_size=output_dimension,
                character_embeddings_size=embedding_size,
                dropout=dropout,
                output_embedding_type=EmbeddingType.Character)

        self.rnn = nn.GRU(embedding_size + hidden_dimension,
                          hidden_dimension, number_of_layers, batch_first=True)

        self.attention = attention

        self.fc_out = nn.Linear(
            embedding_size + hidden_dimension * 2, output_dimension)
        self.dropout = nn.Dropout(dropout)

    @overrides
    def forward(self, input_sequence, hidden, encoder_context):
        input_batch = SimpleBatchRepresentation(
            device=self._device,
            batch_size=input_sequence.shape[0],
            sequences=input_sequence.unsqueeze(1))

        embeddings = self._embedding_layer.forward(input_batch)

        attention_result = self.attention(hidden, encoder_context)
        attention_result = attention_result.unsqueeze(1)

        weighted_context = torch.bmm(attention_result, encoder_context)
        rnn_input = torch.cat((embeddings, weighted_context), dim=2)
        output, hidden = self.rnn.forward(rnn_input, hidden)

        output = torch.cat(
            (embeddings.squeeze(1), hidden.squeeze(0), encoder_context.squeeze(1)), dim=1)

        prediction = self.fc_out(output)
        return prediction, hidden
