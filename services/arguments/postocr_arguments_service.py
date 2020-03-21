import argparse

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService

from enums.configuration import Configuration


class PostOCRArgumentsService(PretrainedArgumentsService):
    def __init__(self):
        super().__init__()

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

    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        super()._add_specific_arguments(parser)

        parser.add_argument('--encoder-embedding-size', type=int, default=128,
                            help='The size used for generating embeddings in the encoder')
        parser.add_argument('--decoder-embedding-size', type=int, default=16,
                            help='The size used for generating embeddings in the decoder')
        parser.add_argument('--hidden-dimension', type=int, default=256,
                            help='The dimension size used for hidden layers')
        parser.add_argument('--dropout', type=float, default=0.0,
                            help='Dropout probability')
        parser.add_argument('--number-of-layers', type=int, default=1,
                            help='Number of layers used for RNN or Transformer models')
        parser.add_argument('--max-articles-length', type=int, default=1000,
                            help='This is the maximum length of articles that will be used in models. Articles longer than this length will be cut.')

        parser.add_argument('--teacher-forcing-ratio', type=float, default=0.5,
                            help='Ratio for teacher forcing during decoding of translation. Default is 0.5')

    def _validate_arguments(self, parser: argparse.ArgumentParser):
        super()._validate_arguments(parser)

        teacher_forcing_ratio = self._arguments['teacher_forcing_ratio']
        if teacher_forcing_ratio < 0 or teacher_forcing_ratio > 1:
            raise parser.error('"--teacher-forcing-ratio" must be a value between 0.0 and 1.0')


    @property
    def encoder_embedding_size(self) -> int:
        return self._get_argument('encoder_embedding_size')

    @property
    def decoder_embedding_size(self) -> int:
        return self._get_argument('decoder_embedding_size')

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
    def teacher_forcing_ratio(self) -> float:
        return self._get_argument('teacher_forcing_ratio')

    @property
    def max_articles_length(self) -> int:
        return self._get_argument('max_articles_length')
