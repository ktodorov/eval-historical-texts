from datetime import datetime, timedelta

from termcolor import colored

from entities.metric import Metric

class LogService:
    def __init__(self):
        self._log_header = '  Time Epoch Iteration   Progress  (%Epoch) | Train Loss Train Accuracy | Validation Loss Val. Accuracy | Best'
        self._log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,| {:>10.6f} {:>14.10f} | {:>15.11f} {:>13.9f} | {:>4s}'.split(','))

        self._start_time = datetime.now()

        self._progress_color = 'red'
        self._evaluation_color = 'cyan'

    def log_progress(
            self,
            current_step: int,
            all_steps: int,
            evaluation: bool = False):

        prefix = 'Train'
        if evaluation:
            prefix = 'Evaluating'

        print(colored(f'{prefix}: {current_step}/{all_steps}       \r', self._progress_color), end='')

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
        print(colored(
            self._log_template.format(
                time_passed.total_seconds(),
                epoch,
                iteration,
                1 + iteration,
                iterations,
                100. * (1 + iteration) / iterations,
                train_metric.get_current_loss(),
                train_metric.get_current_accuracy(),
                validation_metric.get_current_loss(),
                validation_metric.get_current_accuracy(),
                "BEST" if new_best else ""), self._evaluation_color))

        # self.writer.add_scalar(
        #     "Loss_validation", loss_validation, batches_done, time_passed)
        # self.writer.add_scalar("Loss_train", loss_train,
        #                        batches_done, time_passed)

    def get_time_passed(self) -> timedelta:
        result = datetime.now() - self._start_time
        return result
