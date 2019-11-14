import argparse

from services.arguments_service_base import ArgumentsServiceBase


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
        parser.add_argument("--shuffle", action='store_true',
                            help="shuffle datasets while training")
        parser.add_argument("--learning-rate", type=float, default=2e-5,
                            help="learning rate for training models")

        # Transformer specific settings
        parser.add_argument('--pretrained-weights', type=str, default='bert-base-cased',
                            help='weights to use for initializing transformer models')
        parser.add_argument('--configuration', type=str, default='kbert',
                            help='Which configuration of model to load and use')
