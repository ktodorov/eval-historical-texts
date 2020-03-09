import argparse

from services.arguments.arguments_service_base import ArgumentsServiceBase

from enums.metric_type import MetricType
from enums.configuration import Configuration


class PretrainedArgumentsService(ArgumentsServiceBase):
    def __init__(self):
        super().__init__()

    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        super()._add_specific_arguments(parser)

        parser.add_argument('--pretrained-weights', type=str, default='bert-base-cased',
                            help='weights to use for initializing transformer models')
        parser.add_argument('--include-pretrained-model', action='store_true',
                            help='Should a pretrained model be used to provide more information')
        parser.add_argument('--pretrained-model-size', type=int, default=768,
                            help='The hidden size dimension of the pretrained model. Default is 768 for BERT')
        parser.add_argument('--pretrained-max-length', type=int, default=None,
                            help='The maximum length the pretrained model(if any). Default is None')
        parser.add_argument('--pretrained-vocabulary-size', type=int, default=1000,
                            help='Vocabulary size of the pretrained representation')
        parser.add_argument('--learn-new-embeddings', action='store_true',
                            help='Whether new embeddings should be learned next to the pretrained representation')

    @property
    def pretrained_weights(self) -> str:
        return self._get_argument('pretrained_weights')

    @property
    def include_pretrained_model(self) -> bool:
        return self._get_argument('include_pretrained_model')

    @property
    def pretrained_model_size(self) -> int:
        return self._get_argument('pretrained_model_size')

    @property
    def pretrained_max_length(self) -> int:
        return self._get_argument('pretrained_max_length')

    @property
    def pretrained_vocabulary_size(self) -> int:
        return self._get_argument('pretrained_vocabulary_size')

    @property
    def learn_new_embeddings(self) -> bool:
        return self._get_argument('learn_new_embeddings')
