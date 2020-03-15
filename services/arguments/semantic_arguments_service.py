import argparse

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService

from enums.threshold_calculation import ThresholdCalculation


class SemanticArgumentsService(PretrainedArgumentsService):
    def __init__(self):
        super().__init__()

    def get_configuration_name(self) -> str:
        result = f'semeval-{str(self.language)}'
        if self.corpus is not None:
            result += f'-{str(self.corpus)}'

        return result

    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        super()._add_specific_arguments(parser)

        parser.add_argument('--word-distance-threshold', type=float, default=None,
                            help='The threshold which will be used to compare against word distance for the SemEval challenge')
        parser.add_argument('--corpus', type=int, default=None,
                            help='The corpus to be used')
        parser.add_argument('--plot-distances', action='store_true',
                            help='Plot distances of target words for the different time periods')
        parser.add_argument('--word-distance-threshold-calculation', type=ThresholdCalculation,
                            choices=list(ThresholdCalculation), default=ThresholdCalculation.Mean,
                            help='Calculation of the threshold in case a constant value is not given')

    @property
    def word_distance_threshold(self) -> float:
        return self._get_argument('word_distance_threshold')

    @property
    def corpus(self) -> int:
        return self._get_argument('corpus')

    @property
    def plot_distances(self) -> bool:
        return self._get_argument('plot_distances')

    @property
    def word_distance_threshold_calculation(self) -> ThresholdCalculation:
        return self._get_argument('word_distance_threshold_calculation')