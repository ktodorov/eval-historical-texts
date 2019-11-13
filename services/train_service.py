import sys
import time
import math
from datetime import datetime
from typing import List, Tuple
import numpy as np

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from losses.loss_base import LossBase
from models.model_base import ModelBase

from services.arguments_service_base import ArgumentsServiceBase
from services.dataloader_service import DataLoaderService


class TrainService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            dataloader_service: DataLoaderService,
            loss_function: LossBase,
            optimizer: Optimizer,
            model: ModelBase):

        self._loss_function = loss_function
        self._optimizer = optimizer
        self._model = model
        self._arguments_service = arguments_service
        self._patience = self._arguments_service.get_argument('patience')

        self.data_loader_train, self.data_loader_validation = dataloader_service.get_train_dataloaders()

        # self._log_header = '  Time Epoch Iteration    Progress (%Epoch) | Train Loss Train Acc. | Valid Loss Valid Acc. | Best | VAE-stuff'
        # self._log_template = ' '.join(
        #     '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,| {:>10.6f} {:>10.6f} | {:>10.6f} {:>10.6f} | {:>4s} | {:>4s}'.split(
        #         ','))
        # self._start_time = time.time()
    def train(self) -> bool:
        """
         main training function
        """

        # setup data output directories:
        # setup_directories()
        # save_codebase_of_run(self.arguments)

        # data gathering
        progress = []

        epoch = 0

        try:

            # print(self._log_header)

            best_metrics = (math.inf, 0)
            patience = self._patience
            # run
            for epoch in range(self._arguments_service.get_argument('epochs')):
                # do epoch
                epoch_progress, best_metrics, patience = self._epoch_iteration(
                    epoch, best_metrics, patience)

                # add progress-list to global progress-list
                progress += epoch_progress

                # write progress to pickle file (overwrite because there is no point keeping seperate versions)
                # DATA_MANAGER.save_python_obj(progress,
                #                              os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, PROGRESS_DIR,
                #                                           "progress_list"),
                #                              print_success=False)

                # flush prints
                sys.stdout.flush()

                if patience == 0:
                    break

        except KeyboardInterrupt as e:
            print(f"Killed by user: {e}")
            # save_models([self._model], f"KILLED_at_epoch_{epoch}")
            return False
        except Exception as e:
            print(e)
            # save_models([self._model], f"CRASH_at_epoch_{epoch}")
            raise e

        # flush prints
        sys.stdout.flush()

        # example last save
        # save_models([self._model], "finished")
        return True

    def _epoch_iteration(
            self,
            epoch_num: int,
            best_metrics: Tuple[float, float],
            patience: int) -> Tuple[List, Tuple, int]:
        """
        one epoch implementation
        """

        # if not self.arguments.train_classifier:
        #     self._loss_function.reset()

        progress = []

        train_accuracy = 0
        train_loss = 0
        data_loader_length = len(self.data_loader_train)

        for i, (batch, targets, lengths) in enumerate(self.data_loader_train):
            print(f'Train: {i}/{data_loader_length}       \r', end='')

            # do forward pass and whatnot on batch
            loss_batch, accuracy_batch = self._batch_iteration(
                batch, targets, lengths, i)
            train_loss += loss_batch
            train_accuracy += accuracy_batch

            # add to list somehow:
            progress.append({"loss": loss_batch, "acc": accuracy_batch})

            # calculate amount of batches and walltime passed
            batches_passed = i + (epoch_num * len(self.data_loader_train))
            time_passed = datetime.now() - DATA_MANAGER.actual_date

            # run on validation set and print progress to terminal
            # if we have eval_frequency or if we have finished the epoch
            if (batches_passed % self._arguments_service.get_argument('eval_freq')) == 0 or (i + 1 == data_loader_length):
                loss_validation, acc_validation = self._evaluate()

                new_best = False
                if self._model.compare_metric(best_metrics, loss_validation, acc_validation):
                    # save_models([self._model], 'model_best')
                    best_metrics = (loss_validation, acc_validation)
                    new_best = True
                    patience = self._patience
                else:
                    patience -= 1

                self._log(
                    loss_validation,
                    acc_validation,
                    (train_loss / (i + 1)),
                    (train_accuracy / (i + 1)),
                    batches_passed,
                    float(time_passed.microseconds),
                    epoch_num,
                    i,
                    data_loader_length,
                    new_best)

            # check if runtime is expired
            if (time_passed.total_seconds() > (self._arguments_service.get_argument('max_training_minutes') * 60)) \
                    and self._arguments_service.get_argument('max_training_minutes') > 0:
                raise KeyboardInterrupt(f"Process killed because {self._arguments_service.get_argument('max_training_minutes')} minutes passed "
                                        f"since {DATA_MANAGER.actual_date}. Time now is {datetime.now()}")

            if patience == 0:
                break

        return progress, best_metrics, patience

    def _batch_iteration(self,
                         batch: torch.Tensor,
                         targets: torch.Tensor,
                         lengths: torch.Tensor,
                         step: int,
                         train_mode: bool = True) -> Tuple[float, float]:
        """
        runs forward pass on batch and backward pass if in train_mode
        """

        batch = batch.to(self._device)
        targets = targets.to(self._device)
        lengths = lengths.to(self._device)

        if train_mode:
            self._model.train()
            self._optimizer.zero_grad()
        else:
            self._model.eval()

        output = self._model.forward(
            batch, lengths=lengths, step=step, label=targets)
        loss = self._loss_function.forward(targets, *output)

        accuracy = 0
        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(), max_norm=5.0)
            self._optimizer.step()

        # if self.arguments.train_classifier:
            accuracy = self._model.calculate_accuracy(targets, *output).item()

        return loss.item(), accuracy

    def _evaluate(self) -> Tuple[float, float]:
        """
        runs iteration on validation set
        """

        accuracies = []
        losses = []
        data_loader_length = len(self.data_loader_validation)

        for i, (batch, targets, lengths) in enumerate(self.data_loader_validation):
            print(f'Validation: {i}/{data_loader_length}       \r', end='')

            # do forward pass and whatnot on batch
            loss_batch, accuracy_batch = self._batch_iteration(
                batch, targets, lengths, i, train_mode=False)
            accuracies.append(accuracy_batch)
            losses.append(loss_batch)

        return float(np.mean(losses)), float(np.mean(accuracies))

    def _log(self,
             loss_validation: float,
             acc_validation: float,
             loss_train: float,
             acc_train: float,
             batches_done: int,
             time_passed: float,
             epoch: int,
             iteration: int,
             iterations: int,
             new_best: bool):
        """
        logs progress to user through tensorboard and terminal
        """
        pass
        # self.writer.add_scalar("Accuracy_validation",
        #                        acc_validation, batches_done, time_passed)
        # self.writer.add_scalar("Accuracy_train", acc_train,
        #                        batches_done, time_passed)

        # print(self._log_template.format(
        #     time.time() - self._start_time,
        #     epoch,
        #     iteration,
        #     1 + iteration,
        #     iterations,
        #     100. * (1 + iteration) / iterations,
        #     loss_train,
        #     acc_train,
        #     loss_validation,
        #     acc_validation,
        #     "BEST" if new_best else ""
        # ))

        # self.writer.add_scalar(
        #     "Loss_validation", loss_validation, batches_done, time_passed)
        # self.writer.add_scalar("Loss_train", loss_train,
        #                        batches_done, time_passed)
