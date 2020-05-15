from datetime import datetime, timedelta
from termcolor import colored
import wandb
import torch
import numpy as np

from entities.metric import Metric
from entities.data_output_log import DataOutputLog
from services.arguments.arguments_service_base import ArgumentsServiceBase


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
                project=str(arguments_service.challenge),
                config=arguments_service._arguments,
                entity='eval-historical-texts',
                force=True,
                name=arguments_service.get_configuration_name()
                # resume=arguments_service.resume_training,
                # id='' #TODO
            )

    def log_progress(
            self,
            current_step: int,
            all_steps: int,
            epoch_num: int = None,
            evaluation: bool = False):

        prefix = 'Train'
        if evaluation:
            prefix = 'Evaluating'
        else:
            self.log_summary('Iteration', current_step)

        epoch_str = 'N/A'
        if epoch_num is not None:
            epoch_str = str(epoch_num)

        print(colored(
            f'{prefix}: {current_step}/{all_steps}       | Epoch: {epoch_str}           \r', self._progress_color), end='')

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
            new_best: bool,
            metric_log_key: str = None):
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
        if train_accuracies and len(train_accuracies) > 0:
            if metric_log_key is not None and train_metric.contains_accuracy_metric(metric_log_key):
                train_accuracy = train_metric.get_accuracy_metric(
                    metric_log_key)
            else:
                train_accuracy = list(train_accuracies.values())[0]
        else:
            train_accuracy = 0

        if validation_accuracies and len(validation_accuracies) > 0:
            if metric_log_key is not None and validation_metric.contains_accuracy_metric(metric_log_key):
                validation_accuracy = validation_metric.get_accuracy_metric(
                    metric_log_key)
            else:
                validation_accuracy = list(validation_accuracies.values())[0]
        else:
            validation_accuracy = 0

        print(colored(
            self._log_template.format(
                time_passed.total_seconds(),
                epoch,
                iteration,
                1 + iteration,
                iterations,
                100. * (1 + iteration) / iterations,
                train_loss,
                train_accuracy,
                validation_loss,
                validation_accuracy,
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

    def log_batch_results(self, data_output_log: DataOutputLog):
        if not self._external_logging_enabled or data_output_log is None:
            return

        columns, data = data_output_log.get_log_data()
        table_log = wandb.Table(columns=columns, data=data)

        wandb.log({
            'batch results': table_log
        }, step=self._get_current_step())

    def log_incremental_metric(self, metric_key: str, metric_value: object):
        if not self._external_logging_enabled:
            return

        wandb.log({
            metric_key: metric_value
        }, step=self._get_current_step())

    def log_heatmap(
            self,
            heatmap_title: str,
            matrix_values: np.array,
            x_labels: list,
            y_labels: list,
            show_text_inside: bool = False):
        if not self._external_logging_enabled:
            return

        wandb.log({
            heatmap_title: wandb.plots.HeatMap(
                x_labels,
                y_labels,
                matrix_values,
                show_text=show_text_inside)
        }, step=self._get_current_step())

    def start_logging_model(self, model: torch.nn.Module, criterion: torch.nn.Module = None):
        if not self._external_logging_enabled:
            return

        wandb.watch(model, criterion=criterion)

    def get_time_passed(self) -> timedelta:
        result = datetime.now() - self._start_time
        return result

    def _get_current_step(self) -> int:
        return (self._current_epoch * self._all_iterations) + self._current_iteration
