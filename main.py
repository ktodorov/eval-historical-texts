import numpy as np
import torch
import random

from transformers import *

from services.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService
from services.train_service import TrainService
from services.test_service import TestService


def main(
        data_service: DataService,
        arguments_service: ArgumentsServiceBase,
        train_service: TrainService,
        test_service: TestService):

    # print the arguments that the program was initialized with
    arguments_service.print_arguments()

    if arguments_service.evaluate:
        test_service.test()
    else:
        train_service.train()