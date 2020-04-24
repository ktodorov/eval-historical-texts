import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from overrides import overrides

from transformers import BertModel

from enums.embedding_type import EmbeddingType
from entities.batch_representation import BatchRepresentation

from models.embedding.embedding_layer import EmbeddingLayer

from services.pretrained_representations_service import PretrainedRepresentationsService


class SequenceEncoder(nn.Module):
    def __init__(
            self,
            pretrained_representations_service: PretrainedRepresentationsService,
            device: str,
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
            shared_embedding_layer: EmbeddingLayer = None):
        super().__init__()

        assert learn_embeddings or include_pretrained

        self._pretrained_representations_service = pretrained_representations_service
        self._include_pretrained = include_pretrained
        additional_size = pretrained_hidden_size if self._include_pretrained else 0
        self._learn_embeddings = learn_embeddings
        self._bidirectional = bidirectional
        self._use_own_embeddings = use_own_embeddings
        if not self._use_own_embeddings:
            if shared_embedding_layer is None:
                raise Exception('Shared embedding layer not supplied')

            self._embedding_layer = shared_embedding_layer
        else:
            self._embedding_layer = EmbeddingLayer(
                pretrained_representations_service,
                device=device,
                learn_character_embeddings=learn_embeddings,
                include_pretrained_model=include_pretrained,
                pretrained_model_size=pretrained_hidden_size,
                vocabulary_size=input_size,
                character_embeddings_size=embedding_size,
                dropout=dropout,
                output_embedding_type=EmbeddingType.Character)
            # lstm_input_size = additional_size
            # if learn_embeddings:
            #     if self._use_own_embeddings:
            #         self.embedding = nn.Embedding(input_size, embedding_size)

            #     lstm_input_size += embedding_size

        self.rnn = nn.GRU(self._embedding_layer.output_size, hidden_dimension,
                          number_of_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    @overrides
    def forward(self, input_batch: BatchRepresentation, debug: bool = False, **kwargs):
        embeddings = self._embedding_layer.forward(input_batch)
        # if self._learn_embeddings:
        #     if self._use_own_embeddings:
        #         embedded = self.embedding(input_batch)
        #     else:
        #         embedded = self._shared_embeddings(input_batch)

        #     embedded = self.dropout(embedded)

        #     if self._include_pretrained:
        #         embedded = self._pretrained_representations_service.add_pretrained_representations_to_character_embeddings(
        #             embedded, pretrained_representations, offset_lists)
        # else:
        #     embedded = pretrained_representations

        x_packed = pack_padded_sequence(embeddings, input_batch.lengths, batch_first=True)

        _, hidden = self.rnn.forward(x_packed)

        if self._bidirectional:
            hidden = torch.cat(
                (hidden[0, :, :], hidden[1, :, :]), dim=1).unsqueeze(0)

        return hidden
