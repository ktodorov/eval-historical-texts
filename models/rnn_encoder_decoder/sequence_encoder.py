import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from overrides import overrides

from transformers import BertModel

from enums.embedding_type import EmbeddingType
from entities.batch_representation import BatchRepresentation
from entities.options.embedding_layer_options import EmbeddingLayerOptions
from entities.options.pretrained_representations_options import PretrainedRepresentationsOptions

from models.embedding.embedding_layer import EmbeddingLayer

from services.file_service import FileService
from models.model_base import ModelBase

class SequenceEncoder(ModelBase):
    def __init__(
            self,
            file_service: FileService,
            device: str,
            pretrained_representations_options: PretrainedRepresentationsOptions,
            embedding_size: int,
            input_size: int,
            hidden_dimension: int,
            number_of_layers: int,
            dropout: float = 0,
            learn_embeddings: bool = True,
            bidirectional: bool = False,
            use_own_embeddings: bool = True,
            shared_embedding_layer: EmbeddingLayer = None):
        super().__init__()

        self._bidirectional = bidirectional
        self._use_own_embeddings = use_own_embeddings
        if not self._use_own_embeddings:
            if shared_embedding_layer is None:
                raise Exception('Shared embedding layer not supplied')

            self._embedding_layer = shared_embedding_layer
        else:
            embedding_layer_options = EmbeddingLayerOptions(
                device=device,
                pretrained_representations_options=pretrained_representations_options,
                learn_character_embeddings=learn_embeddings,
                vocabulary_size=input_size,
                character_embeddings_size=embedding_size,
                dropout=dropout,
                output_embedding_type=EmbeddingType.Character)

            self._embedding_layer = EmbeddingLayer(file_service, embedding_layer_options)

        self.rnn = nn.GRU(self._embedding_layer.output_size, hidden_dimension,
                          number_of_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    @overrides
    def forward(self, input_batch: BatchRepresentation, debug: bool = False, **kwargs):
        embeddings = self._embedding_layer.forward(input_batch)

        x_packed = pack_padded_sequence(
            embeddings, input_batch.lengths, batch_first=True)

        _, hidden = self.rnn.forward(x_packed)

        if self._bidirectional:
            hidden = torch.cat(
                (hidden[0, :, :], hidden[1, :, :]), dim=1).unsqueeze(0)

        return hidden
