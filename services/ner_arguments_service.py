import argparse

from services.pretrained_arguments_service import PretrainedArgumentsService

from enums.metric_type import MetricType
from enums.configuration import Configuration


class NERArgumentsService(PretrainedArgumentsService):
    def __init__(self):
        super().__init__()

    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        super()._add_specific_arguments(parser)

        parser.add_argument('--embeddings-size', type=int, default=128,
                            help='The size used for generating embeddings in the encoder')
        parser.add_argument('--hidden-dimension', type=int, default=256,
                            help='The dimension size used for hidden layers')
        parser.add_argument('--dropout', type=float, default=0.0,
                            help='Dropout probability')
        parser.add_argument('--number-of-layers', type=int, default=1,
                            help='Number of layers used for the RNN')

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