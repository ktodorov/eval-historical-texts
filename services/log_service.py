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
            arguments_service: ArgumentsServiceBase,
            external_logging_enabled: bool = False):
        self._log_header = '  Time Epoch Iteration   Progress  (%Epoch) | Train Loss Train Accuracy | Validation Loss Val. Accuracy | Best'
        self._log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,| {:>10.6f} {:>14.10f} | {:>15.11f} {:>13.9f} | {:>4s}'.split(','))

        self._start_time = datetime.now()

        self._progress_color = 'red'
        self._evaluation_color = 'cyan'

        self._current_epoch = 0
        self._all_iterations = 0
        self._current_iteration = 0

        self._external_logging_enabled = external_logging_enabled
        if self._external_logging_enabled:
            wandb.init(
                project='default',
                config=arguments_service._arguments,
                entity='eval-historical-texts',
                force=True
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
        else:
            self.log_summary('Iteration', current_step)

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

        self._current_epoch = epoch
        self._current_iteration = iteration
        self._all_iterations = iterations

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
                list(train_accuracies.values())[0],
                validation_loss,
                list(validation_accuracies.values())[0],
                "BEST" if new_best else ""), self._evaluation_color))

        if self._external_logging_enabled:
            current_step = self._get_current_step()
            wandb.log({'Train loss': train_loss},
                      step=current_step)

            for key, value in train_accuracies.items():
                wandb.log({f'Train - {key}': value},
                          step=current_step)

            for key, value in validation_accuracies.items():
                wandb.log({f'Validation - {key}': value},
                          step=current_step)

            wandb.log({'Validation loss': validation_loss},
                      step=current_step)

            if current_step == 0:
                seconds_per_iteration = time_passed.total_seconds()
            else:
                seconds_per_iteration = time_passed.total_seconds() / current_step

            self.log_summary('Seconds per iteration', seconds_per_iteration)

    def log_summary(self, key: str, value: object):
        if not self._external_logging_enabled:
            return

        wandb.run.summary[key] = value

    def log_batch_results(self, input: str, output: str, expected: str):
        if not self._external_logging_enabled:
            return

        table_log = wandb.Table(data=[[input, output, expected]])
        time_passed = self.get_time_passed()

        wandb.log({'batch results': table_log}, step=self._get_current_step())

    def start_logging_model(self, model: torch.nn.Module, criterion: torch.nn.Module = None):
        if not self._external_logging_enabled:
            return

        wandb.watch(model, criterion=criterion)

    def get_time_passed(self) -> timedelta:
        result = datetime.now() - self._start_time
        return result

    def _get_current_step(self) -> int:
        return (self._current_epoch * self._all_iterations) + self._current_iteration
