import argparse

from services.pretrained_arguments_service import PretrainedArgumentsService



class SemanticArgumentsService(PretrainedArgumentsService):
    def __init__(self):
        super().__init__()

    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        super()._add_specific_arguments(parser)

        parser.add_argument('--word-distance-threshold', type=float, default=100.0,
                            help='The threshold which will be used to compare against word distance for the SemEval challenge')
        parser.add_argument('--corpus', type=int, required=True,
                            help='The corpus to be used')



    @property
    def word_distance_threshold(self) -> float:
        return self._get_argument('word_distance_threshold')

    @property
    def corpus(self) -> int:
        return self._get_argument('corpus')
