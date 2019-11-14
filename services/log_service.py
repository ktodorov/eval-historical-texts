from datetime import datetime, timedelta

from termcolor import colored

class LogService:
    def __init__(self):
        self._log_header = '  Time Epoch Iteration   Progress  (%Epoch) | Train Loss 1 | Train Loss 2 | Best'
        self._log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,| {:>12.8f} | {:>12.8f} | {:>4s}'.split(','))

        self._start_time = datetime.now()

        self._progress_color = 'red'
        self._evaluation_color = 'cyan'

    def log_progress(
            self,
            current_step: int,
            all_steps: int):
        print(colored(f'Train: {current_step}/{all_steps}       \r', self._progress_color), end='')

    def initialize_evaluation(self):
        print(self._log_header)

    def log_evaluation(
            self,
            loss_train: float,
            batches_done: int,
            epoch: int,
            iteration: int,
            iterations: int,
            new_best: bool):
        """
        logs progress to user through tensorboard and terminal
        """
        # self.writer.add_scalar("Accuracy_validation",
        #                        acc_validation, batches_done, time_passed)
        # self.writer.add_scalar("Accuracy_train", acc_train,
        #                        batches_done, time_passed)

        time_passed = self.get_time_passed()
        print(colored(
            self._log_template.format(
                time_passed.total_seconds(),
                epoch,
                iteration,
                1 + iteration,
                iterations,
                100. * (1 + iteration) / iterations,
                loss_train[0] if isinstance(loss_train, list) else loss_train,
                loss_train[1] if isinstance(loss_train, list) else -1,
                "BEST" if new_best else ""), self._evaluation_color))

        # self.writer.add_scalar(
        #     "Loss_validation", loss_validation, batches_done, time_passed)
        # self.writer.add_scalar("Loss_train", loss_train,
        #                        batches_done, time_passed)

    def get_time_passed(self) -> timedelta:
        result = datetime.now() - self._start_time
        return result
