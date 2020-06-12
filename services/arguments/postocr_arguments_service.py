from overrides import overrides
import argparse

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService

from enums.configuration import Configuration


class PostOCRArgumentsService(PretrainedArgumentsService):
    def __init__(self):
        super().__init__()

    @overrides
    def get_configuration_name(self) -> str:
        result = str(self.configuration)
        if self.configuration == Configuration.SequenceToCharacter:
            result = 'seq-to-char'
        elif self.configuration == Configuration.TransformerSequence:
            result = 'transformer'

        if self.include_pretrained_model:
            result += '-pretr'

        if not self.learn_new_embeddings:
            result += '-no-emb'

        return result

    @overrides
    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        super()._add_specific_arguments(parser)

        parser.add_argument('--encoder-embedding-size', type=int, default=128,
                            help='The size used for generating embeddings in the encoder')
        parser.add_argument('--decoder-embedding-size', type=int, default=16,
                            help='The size used for generating embeddings in the decoder')
        parser.add_argument('--share-embedding-layer', action='store_true',
                            help='If set to true, the embedding layer of the encoder and decoder will be shared')
        parser.add_argument('--hidden-dimension', type=int, default=256,
                            help='The dimension size used for hidden layers')
        parser.add_argument('--dropout', type=float, default=0.0,
                            help='Dropout probability')
        parser.add_argument('--number-of-layers', type=int, default=1,
                            help='Number of layers used for RNN or Transformer models')
        parser.add_argument('--bidirectional', action='store_true',
                            help='Whether the RNN used will be bidirectional')

        parser.add_argument('--use-beam-search', action='store_true',
                            help='If set to true, beam search will be used for decoding instead of greedy decoding')
        parser.add_argument('--beam-width', type=int, default=3,
                            help='Width of the beam when using beam search. Defaults to 3')

    @overrides
    def _validate_arguments(self, parser: argparse.ArgumentParser):
        super()._validate_arguments(parser)

    @property
    def encoder_embedding_size(self) -> int:
        return self._get_argument('encoder_embedding_size')

    @property
    def decoder_embedding_size(self) -> int:
        return self._get_argument('decoder_embedding_size')

    @property
    def share_embedding_layer(self) -> bool:
        return self._get_argument('share_embedding_layer')

    @property
    def hidden_dimension(self) -> int:
        return self._get_argument('hidden_dimension')

    @property
    def dropout(self) -> float:
        return self._get_argument('dropout')

    @property
    def number_of_layers(self) -> int:
        return self._get_argument('number_of_layers')

    @property
    def bidirectional(self) -> bool:
        return self._get_argument('bidirectional')

    @property
    def use_beam_search(self) -> bool:
        return self._get_argument('use_beam_search')

    @property
    def beam_width(self) -> int:
        return self._get_argument('beam_width')
