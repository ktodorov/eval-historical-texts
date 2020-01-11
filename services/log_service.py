from datetime import datetime, timedelta
from termcolor import colored
import wandb
import torch
import numpy as np

from entities.metric import Metric
from services.arguments_service_base import ArgumentsServiceBase


class LogService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase):
        self._log_header = '  Time Epoch Iteration   Progress  (%Epoch) | Train Loss Train Accuracy | Validation Loss Val. Accuracy | Best'
        self._log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,| {:>10.6f} {:>14.10f} | {:>15.11f} {:>13.9f} | {:>4s}'.split(','))

        self._start_time = datetime.now()

        self._progress_color = 'red'
        self._evaluation_color = 'cyan'

        wandb.init(
            project='eval-historical-texts',
            config=arguments_service._arguments
            # resume=arguments_service.get_argument('resume_training'),
            # id='' #TODO
        )

    def log_progress(
            self,
            current_step: int,
            all_steps: int,
            evaluation: bool = False):

        prefix = 'Train'
        if evaluation:
            prefix = 'Evaluating'

        print(colored(
            f'{prefix}: {current_step}/{all_steps}       \r', self._progress_color), end='')

    def initialize_evaluation(self):
        print(self._log_header)

    def log_evaluation(
            self,
            train_metric: Metric,
            validation_metric: Metric,
            batches_done: int,
            epoch: int,
            iteration: int,
            iterations: int,
            new_best: bool):
        """
        logs progress to user through tensorboard and terminal
        """

        time_passed = self.get_time_passed()
        train_loss = train_metric.get_current_loss()
        train_accuracies = train_metric.get_current_accuracies()
        validation_loss = validation_metric.get_current_loss()
        validation_accuracies = validation_metric.get_current_accuracies()

        print(colored(
            self._log_template.format(
                time_passed.total_seconds(),
                epoch,
                iteration,
                1 + iteration,
                iterations,
                100. * (1 + iteration) / iterations,
                train_loss,
                np.mean([*train_accuracies.values()]),
                validation_loss,
                np.mean([*validation_accuracies.values()]),
                "BEST" if new_best else ""), self._evaluation_color))

        wandb.log({'Train loss': train_loss},
                  step=time_passed.seconds, commit=False)

        for key, value in train_accuracies.items():
            wandb.log({f'Train accuracy - {key}': value},
                      step=time_passed.seconds, commit=False)

        for key, value in validation_accuracies.items():
            wandb.log({f'Validation accuracy - {key}': value},
                      step=time_passed.seconds, commit=False)

        wandb.log({'Validation loss': validation_loss},
                  step=time_passed.seconds)

    def log_summary(self, key: str, value: object):
        wandb.run.summary[key] = value

    def start_logging_model(self, model: torch.nn.Module, criterion: torch.nn.Module = None):
        wandb.watch(model, criterion=criterion)

    def get_time_passed(self) -> timedelta:
        result = datetime.now() - self._start_time
        return result
