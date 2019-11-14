import numpy as np
import torch
import random

from transformers import *

from services.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService
from services.train_service import TrainService


def main(
        data_service: DataService,
        arguments_service: ArgumentsServiceBase,
        train_service: TrainService):

    initialize_seed(
        arguments_service.get_argument('seed'),
        arguments_service.get_argument('device'))

    if arguments_service.get_argument('evaluate'):
        pass
    else:
        train_service.train()


def initialize_seed(seed: int, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
