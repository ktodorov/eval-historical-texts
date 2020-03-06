import argparse

from typing import List, Dict

from entities.custom_argument_parser import CustomArgumentParser

from enums.evaluation_type import EvaluationType
from enums.language import Language
from enums.output_format import OutputFormat
from enums.challenge import Challenge
from enums.configuration import Configuration
from enums.metric_type import MetricType
from enums.experiment_type import ExperimentType


class ArgumentsServiceBase:
    def __init__(self, raise_errors_on_invalid_args: bool = True):
        self._raise_errors_on_invalid_args = raise_errors_on_invalid_args
        self._arguments: argparse.Namespace = {}

        self._parse_arguments()

    def print_arguments(self):
        """Prints the arguments which the program was initialized with
        """
        print(f'Arguments initialized: {self._arguments}')

    def _parse_arguments(self):
        parser = CustomArgumentParser(
            raise_errors_on_invalid_args=self._raise_errors_on_invalid_args)

        self._add_base_arguments(parser)
        self._add_specific_arguments(parser)
        self._arguments: Dict[str, object] = vars(parser.parse_args())
        self._validate_arguments(parser)

    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        pass

    def _add_base_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument('--epochs', default=500,
                            type=int, help='max number of epochs')
        parser.add_argument('--eval-freq', default=50,
                            type=int, help='evaluate every x batches')
        parser.add_argument('--batch-size', default=8,
                            type=int, help='size of batches')
        parser.add_argument('--max-training-minutes', default=24 * 60, type=int,
                            help='max mins of training before save-and-kill')
        parser.add_argument("--device", type=str, default='cuda',
                            help="Device to be used. Pick from cpu/cuda. If default none is used automatic check will be done")
        parser.add_argument("--seed", type=int, default=42,
                            metavar="S", help="random seed (default: 42)")
        parser.add_argument("--evaluate", action='store_true',
                            help="run in evaluation mode")
        parser.add_argument("--patience", type=int, default=30,
                            help="how long will the model wait for improvement before stopping training")
        parser.add_argument("--language", type=Language, choices=list(Language), default=Language.English,
                            help="which language to train on")
        parser.add_argument("--shuffle", action='store_false',
                            help="shuffle datasets while training")
        parser.add_argument("--learning-rate", type=float, default=1e-5,
                            help="learning rate for training models")
        parser.add_argument("--checkpoint-name", type=str, default=None,
                            help="name that can be used to distinguish checkpoints")
        parser.add_argument("--resume-training", action='store_true',
                            help="resume training using saved checkpoints")
        parser.add_argument("--data-folder", type=str, default='data',
                            help='folder where data will be taken from')
        parser.add_argument("--output-folder", type=str, default='results',
                            help='folder where results and checkpoints will be saved')
        parser.add_argument('--checkpoint-folder', type=str, default=None,
                            help='folder where checkpoints will be saved/loaded. If it is not provided, the output folder will be used')
        parser.add_argument('--evaluation-type', type=EvaluationType, choices=list(EvaluationType), nargs='*',
                            help='what type of evaluations should be performed')
        parser.add_argument('--output-eval-format', type=OutputFormat, choices=list(OutputFormat),
                            help='what the format of the output after evaluation will be')
        parser.add_argument("--challenge", type=Challenge, choices=list(Challenge),
                            help='Optional challenge that the model is being trained for. If given, data and output results will be put into a specific folder')
        parser.add_argument('--configuration', type=Configuration, choices=list(Configuration), default=Configuration.KBert,
                            help='Which configuration of model to load and use. Default is kbert')
        parser.add_argument('--metric-types', type=MetricType, choices=list(MetricType), default=MetricType.JaccardSimilarity, nargs='*',
                            help='What metrics should be calculated. Default is only Jaccard similarity')
        parser.add_argument('--joint-model', action='store_true',
                            help='If a joint model should be used instead of a single one')
        parser.add_argument('--joint-model-amount', type=int, default=2,
                            help='How many models should be trained jointly')
        parser.add_argument('--enable-external-logging', action='store_true',
                            help='Should logging to external service be enabled')
        parser.add_argument('--train-dataset-limit-size', type=int, default=None,
                            help='Limit the train dataset. By default no limit is done.')
        parser.add_argument('--validation-dataset-limit-size', type=int, default=None,
                            help='Limit the validation dataset. By default no limit is done.')
        parser.add_argument('--skip-validation', action='store_true',
                            help='Whether validation should be skipped, meaning no validation dataset is loaded and no evaluation is done while training. By default is false')
        parser.add_argument('--run-experiments', action='store_true',
                            help='Whether to run experiments instead of training or evaluation')
        parser.add_argument('--experiment-types', type=ExperimentType, choices=list(ExperimentType), default=None, nargs='*',
                            help='What types of experiments should be run')

    def _validate_arguments(self, parser: argparse.ArgumentParser):
        pass

    def _get_argument(self, key: str) -> object:
        """Returns an argument value from the list of registered arguments

        :param key: key of the argument
        :type key: str
        :raises LookupError: if no argument is found, lookup error will be raised
        :return: the argument value
        :rtype: object
        """
        if key not in self._arguments.keys():
            raise LookupError(f'{key} not found in arguments')

        return self._arguments[key]

    @property
    def epochs(self) -> int:
        return self._get_argument('epochs')

    @property
    def eval_freq(self) -> int:
        return self._get_argument('eval_freq')

    @property
    def batch_size(self) -> int:
        return self._get_argument('batch_size')

    @property
    def max_training_minutes(self) -> int:
        return self._get_argument('max_training_minutes')

    @property
    def device(self) -> str:
        return self._get_argument('device')

    @property
    def seed(self) -> int:
        return self._get_argument('seed')

    @property
    def evaluate(self) -> bool:
        return self._get_argument('evaluate')

    @property
    def patience(self) -> int:
        return self._get_argument('patience')

    @property
    def language(self) -> Language:
        return self._get_argument('language')

    @property
    def shuffle(self) -> bool:
        return self._get_argument('shuffle')

    @property
    def learning_rate(self) -> float:
        return self._get_argument('learning_rate')

    @property
    def checkpoint_name(self) -> str:
        return self._get_argument('checkpoint_name')

    @property
    def resume_training(self) -> bool:
        return self._get_argument('resume_training')

    @property
    def data_folder(self) -> str:
        return self._get_argument('data_folder')

    @property
    def output_folder(self) -> str:
        return self._get_argument('output_folder')

    @property
    def checkpoint_folder(self) -> str:
        return self._get_argument('checkpoint_folder')

    @property
    def evaluation_type(self) -> List[EvaluationType]:
        return self._get_argument('evaluation_type')

    @property
    def output_eval_format(self) -> OutputFormat:
        return self._get_argument('output_eval_format')

    @property
    def challenge(self) -> Challenge:
        return self._get_argument('challenge')

    @property
    def configuration(self) -> Configuration:
        return self._get_argument('configuration')

    @property
    def metric_types(self) -> List[MetricType]:
        return self._get_argument('metric_types')

    @property
    def train_dataset_limit_size(self) -> int:
        return self._get_argument('train_dataset_limit_size')

    @property
    def validation_dataset_limit_size(self) -> int:
        return self._get_argument('validation_dataset_limit_size')

    @property
    def joint_model(self) -> bool:
        return self._get_argument('joint_model')

    @property
    def joint_model_amount(self) -> int:
        return self._get_argument('joint_model_amount')

    @property
    def enable_external_logging(self) -> bool:
        return self._get_argument('enable_external_logging')

    @property
    def skip_validation(self) -> bool:
        return self._get_argument('skip_validation')

    @property
    def run_experiments(self) -> bool:
        return self._get_argument('run_experiments')
        
    @property
    def experiment_types(self) -> List[ExperimentType]:
        return self._get_argument('experiment_types')
