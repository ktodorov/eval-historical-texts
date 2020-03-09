import numpy as np
import torch
import random

from transformers import *

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService
from services.train_service import TrainService
from services.test_service import TestService
from services.experiment_service import ExperimentService


def main(
        data_service: DataService,
        arguments_service: ArgumentsServiceBase,
        train_service: TrainService,
        test_service: TestService,
        experiment_service: ExperimentService):

    # print the arguments that the program was initialized with
    arguments_service.print_arguments()

    if arguments_service.evaluate:
        test_service.test()
    elif not arguments_service.run_experiments:
        train_service.train()
    else:
        experiment_service.execute_experiments(arguments_service.experiment_types)