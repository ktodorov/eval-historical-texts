from overrides import overrides
import argparse

from services.arguments.arguments_service_base import ArgumentsServiceBase

from enums.metric_type import MetricType
from enums.configuration import Configuration
from enums.pretrained_model import PretrainedModel


class PretrainedArgumentsService(ArgumentsServiceBase):
    def __init__(self, raise_errors_on_invalid_args=True):
        super().__init__(raise_errors_on_invalid_args)

    @overrides
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
        parser.add_argument('--learn-new-embeddings', action='store_true',
                            help='Whether new embeddings should be learned next to the pretrained representation')

        parser.add_argument('--fasttext-model', type=str, default=None,
                            help='fasttext model to use for loading additional information')
        parser.add_argument('--include-fasttext-model', action='store_true',
                            help='Should a fasttext model be used to provide more information')
        parser.add_argument('--fasttext-model-size', type=int, default=300,
                            help='The hidden size dimension of the fasttext model. Default is 300')
        parser.add_argument("--pretrained-model", type=PretrainedModel, choices=list(PretrainedModel), default=PretrainedModel.BERT,
                            help="Pretrained model that will be used to tokenize strings and generate embeddings")
        parser.add_argument('--fine-tune-pretrained', action='store_true',
                            help='If true, the loaded pre-trained model will be fine-tuned instead of being frozen. Default is `false`')
        parser.add_argument('--fine-tune-after-convergence', action='store_true',
                            help='If true, the loaded pre-trained model will be fine-tuned but only once the full model has converged. Default is `false`')
        parser.add_argument("--fine-tune-learning-rate", type=float, default=None,
                            help="Different learning rate to use for pre-trained model. If None is given, then the global learning rate will be used. Default is `None`.")

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
    def learn_new_embeddings(self) -> bool:
        return self._get_argument('learn_new_embeddings')

    @property
    def fasttext_model(self) -> str:
        return self._get_argument('fasttext_model')

    @property
    def include_fasttext_model(self) -> bool:
        return self._get_argument('include_fasttext_model')

    @property
    def fasttext_model_size(self) -> int:
        return self._get_argument('fasttext_model_size')

    @property
    def pretrained_model(self) -> PretrainedModel:
        return self._get_argument('pretrained_model')

    @property
    def fine_tune_pretrained(self) -> bool:
        return self._get_argument('fine_tune_pretrained')

    @property
    def fine_tune_after_convergence(self) -> bool:
        return self._get_argument('fine_tune_after_convergence')

    @property
    def fine_tune_learning_rate(self) -> float:
        return self._get_argument('fine_tune_learning_rate')