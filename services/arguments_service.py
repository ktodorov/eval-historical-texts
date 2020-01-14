import argparse

from services.arguments_service_base import ArgumentsServiceBase

from enums.accuracy_type import AccuracyType
from enums.configuration import Configuration
from enums.evaluation_type import EvaluationType
from enums.output_format import OutputFormat


class ArgumentsService(ArgumentsServiceBase):
    def __init__(self):
        super(ArgumentsService, self).__init__()

    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument('--epochs', default=500,
                            type=int, help='max number of epochs')
        parser.add_argument('--eval-freq', default=50,
                            type=int, help='evaluate every x batches')
        parser.add_argument('--batch-size', default=8,
                            type=int, help='size of batches')
        parser.add_argument('--max-training-minutes', default=24 * 60, type=int,
                            help='max mins of training be4 save-and-kill')
        parser.add_argument("--device", type=str, default='cuda',
                            help="Device to be used. Pick from cpu/cuda. "
                            "If default none is used automatic check will be done")
        parser.add_argument("--seed", type=int, default=42,
                            metavar="S", help="random seed (default: 42)")
        parser.add_argument("--evaluate", action='store_true',
                            help="run in evaluation mode")
        parser.add_argument("--patience", type=int, default=30,
                            help="how long will the model wait for improvement before stopping training")
        parser.add_argument("--language", type=str, default='english',
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
        parser.add_argument("--challenge", type=str, default=None,
                            help='Optional challenge that the model is being trained for. If given, data and output results will be put into a specific folder')
        parser.add_argument('--train-dataset-reduction-size', type=float, default=None,
                            help='Proportions to use to reduce the train dataset. Must be a value between 0.0 and 1.0. By default no reduction is done.')
        parser.add_argument('--validation-dataset-reduction-size', type=float, default=None,
                            help='Proportions to use to reduce the validation dataset. Must be a value between 0.0 and 1.0. By default no reduction is done.')
        parser.add_argument('--accuracy-type', type=AccuracyType, choices=list(AccuracyType), default=AccuracyType.CharacterLevel, nargs='*',
                            help='How should the accuracy be calculated. Default is character-level accuracy')
        parser.add_argument('--max-articles-length', type=int, default=1000,
                            help='This is the maximum length of articles that will be used in models. Articles longer than this length will be cut.')
        parser.add_argument('--enable-external-logging', action='store_true',
                            help='Should logging to external service be enabled')

        # Transformer specific settings
        parser.add_argument('--pretrained-weights', type=str, default='bert-base-cased',
                            help='weights to use for initializing transformer models')
        parser.add_argument('--configuration', type=Configuration, choices=list(Configuration), default=Configuration.KBert,
                            help='Which configuration of model to load and use. Default is kbert')
        parser.add_argument('--joint-model', action='store_true',
                            help='If a joint model should be used instead of a single one')
        parser.add_argument('--joint-model-amount', type=int, default=2,
                            help='How many models should be trained jointly')

        # SemEval
        parser.add_argument('--word-distance-threshold', type=float, default=100.0,
                            help='The threshold which will be used to compare against word distance for the SemEval challenge')

        # SentencePiece
        parser.add_argument('--sentence-piece-vocabulary-size', type=int, default=1000,
                            help='Vocabulary size that will be used when training the sentence piece model')

        parser.add_argument('--embedding-size', type=int, default=256,
                            help='The size used for generating embeddings')
        parser.add_argument('--hidden-dimension', type=int, default=256,
                            help='The dimension size used for hidden layers')
        parser.add_argument('--dropout', type=float, default=0.0,
                            help='Dropout probability')
        parser.add_argument('--number-of-layers', type=int, default=1,
                            help='Number of layers used for RNN models')

    def _validate_arguments(self, parser: argparse.ArgumentParser):
        super()._validate_arguments(parser)

        train_dataset_reduction_size = self._arguments['train_dataset_reduction_size']
        if train_dataset_reduction_size and (train_dataset_reduction_size <= 0 or train_dataset_reduction_size >= 1):
            raise parser.error(
                '"--train-dataset-reduction-size" must be between 0.0 and 1.0')

        validation_dataset_reduction_size = self._arguments['validation_dataset_reduction_size']
        if validation_dataset_reduction_size and (validation_dataset_reduction_size <= 0 or validation_dataset_reduction_size >= 1):
            raise parser.error(
                '"--validation-dataset-reduction-size" must be between 0.0 and 1.0')
