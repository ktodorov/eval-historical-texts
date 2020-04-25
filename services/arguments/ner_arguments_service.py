from overrides import overrides

import argparse

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService

from enums.ner_type import NERType


class NERArgumentsService(PretrainedArgumentsService):
    def __init__(self):
        super().__init__()

    @overrides
    def get_configuration_name(self) -> str:
        result = f'ner-{str(self.language)[:2]}'
        result += f'-{str(self.label_type)[:1]}'

        if self.include_pretrained_model:
            result += '-pr'

        if self.include_fasttext_model:
            result += '-ft'

        if self.learn_character_embeddings:
            result += '-ce'

        if not self.learn_new_embeddings:
            result += '-ne'

        return result

    @overrides
    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        super()._add_specific_arguments(parser)

        parser.add_argument('--embeddings-size', type=int, default=128,
                            help='The size used for generating sub-word embeddings')
        parser.add_argument('--hidden-dimension', type=int, default=256,
                            help='The dimension size used for hidden layers')
        parser.add_argument('--dropout', type=float, default=0.0,
                            help='Dropout probability')
        parser.add_argument('--number-of-layers', type=int, default=1,
                            help='Number of layers used for the RNN')
        parser.add_argument('--label-type', type=NERType, choices=list(NERType), default=NERType.Coarse,
                            help='Label type that will be used for classification. Default is Coarse')
        parser.add_argument("--no-attention", action='store_true',
                            help="whether to skip the attention layer")
        parser.add_argument("--bidirectional-rnn", action='store_true',
                            help="whether to use a bidirectional version of the RNN")
        parser.add_argument("--merge-subwords", action='store_true',
                            help="whether to merge the subword embeddings before passing through the RNN")
        parser.add_argument("--learn-character-embeddings", action='store_true',
                            help="whether to learn character embeddings next to the default subword ones")
        parser.add_argument('--character-embeddings-size', type=int, default=32,
                            help='The size used for generating character embeddings')
        parser.add_argument("--replace-all-numbers", action='store_true',
                            help="If all numbers should be replaced by a hard-coded fixed string")

    @property
    def embeddings_size(self) -> int:
        return self._get_argument('embeddings_size')

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
    def label_type(self) -> NERType:
        return self._get_argument('label_type')

    @property
    def use_attention(self) -> bool:
        return not self._get_argument('no_attention')

    @property
    def bidirectional_rnn(self) -> bool:
        return not self._get_argument('bidirectional_rnn')

    @property
    def merge_subwords(self) -> bool:
        return self._get_argument('merge_subwords')

    @property
    def learn_character_embeddings(self) -> bool:
        return self._get_argument('learn_character_embeddings')

    @property
    def character_embeddings_size(self) -> int:
        return self._get_argument('character_embeddings_size')

    @property
    def replace_all_numbers(self) -> int:
        return self._get_argument('replace_all_numbers')