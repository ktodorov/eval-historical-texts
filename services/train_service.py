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
from optimizers.optimizer_base import OptimizerBase

from services.arguments_service_base import ArgumentsServiceBase
from services.dataloader_service import DataLoaderService
from services.log_service import LogService

from transformers import BertTokenizer


class TrainService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            dataloader_service: DataLoaderService,
            loss_function: LossBase,
            optimizer: OptimizerBase,
            log_service: LogService,
            model: ModelBase):

        self._arguments_service = arguments_service
        self._log_service = log_service

        self._loss_function = loss_function
        self._optimizer = optimizer
        self._model = model.to(arguments_service.get_argument('device'))
        self._patience = self._arguments_service.get_argument('patience')

        (self.data_loader_train,
         self.data_loader_validation) = dataloader_service.get_train_dataloaders()

    def train(self) -> bool:
        """
         main training function
        """

        # setup data output directories:
        # setup_directories()
        # save_codebase_of_run(self.arguments)

        epoch = 0

        try:
            self._log_service.initialize_evaluation()

            best_metrics = None
            patience = self._patience
            losses: List[float] = []

            # run
            for epoch in range(self._arguments_service.get_argument('epochs')):

                best_metrics, patience = self._perform_epoch_iteration(
                    epoch, best_metrics, patience, losses)

                # write progress to pickle file (overwrite because there is no
                # point keeping seperate versions)
                # DATA_MANAGER.save_python_obj(progress,
                #                              os.path.join(RESULTS_DIR,
                #                              DATA_MANAGER.stamp,
                #                              PROGRESS_DIR,
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

    def _perform_epoch_iteration(
            self,
            epoch_num: int,
            best_metrics: float,
            patience: int,
            losses: List[float]) -> Tuple[float, int, float]:
        """
        one epoch implementation
        """
        train_accuracy = 0.0
        train_loss = 0.0
        data_loader_length = len(self.data_loader_train)

        for i, batch in enumerate(self.data_loader_train):
            self._log_service.log_progress(i, data_loader_length)

            loss_batch, accuracy_batch = self._perform_batch_iteration(batch)
            losses.append(loss_batch)
            train_loss = np.mean(losses, axis=0)
            train_accuracy += accuracy_batch

            # calculate amount of batches and walltime passed
            batches_passed = i + (epoch_num * len(self.data_loader_train))

            # run on validation set and print progress to terminal
            # if we have eval_frequency or if we have finished the epoch
            if (batches_passed % self._arguments_service.get_argument('eval_freq')) == 0 or (i + 1 == data_loader_length):
                # loss_validation, acc_validation = self._evaluate()

                new_best = self._model.compare_metric(best_metrics, train_loss)
                if new_best:
                    # save_models([self._model], 'model_best')
                    best_metrics = train_loss
                    patience = self._patience
                else:
                    patience -= 1

                self._log_service.log_evaluation(
                    train_loss,
                    batches_passed,
                    epoch_num,
                    i,
                    data_loader_length,
                    new_best)

            # check if runtime is expired
            time_passed = self._log_service.get_time_passed()
            if ((time_passed.total_seconds() > (self._arguments_service.get_argument('max_training_minutes') * 60)) and
                    self._arguments_service.get_argument('max_training_minutes') > 0):
                raise KeyboardInterrupt(
                    f"Process killed because {self._arguments_service.get_argument('max_training_minutes')} minutes passed")

            if patience == 0:
                break

        return best_metrics, patience

    def _perform_batch_iteration(
            self,
            batch: torch.Tensor,
            train_mode: bool = True) -> Tuple[float, float]:
        """
        runs forward pass on batch and backward pass if in train_mode
        """

        if train_mode:
            self._model.train()
            self._optimizer.zero_grad()
        else:
            self._model.eval()

        outputs = self._model.forward(batch)

        accuracy = 0
        if train_mode:
            loss = self._loss_function.backward(outputs)
            self._model.clip_gradients()
            self._optimizer.step()
            self._model.zero_grad()
        else:
            loss = self._loss_function.calculate_loss(outputs)

        # accuracy = self._model.calculate_accuracy(targets, *output).item()

        return loss, accuracy

    # def _evaluate(self) -> Tuple[float, float]:
    #     """
    #     runs iteration on validation set
    #     """

    #     accuracies = []
    #     losses = []
    #     data_loader_length = len(self.data_loader_validation)

    #     for i, (batch, targets, lengths) in enumerate(self.data_loader_validation):
    #         print(f'Validation: {i}/{data_loader_length}       \r', end='')

    #         # do forward pass and whatnot on batch
    #         loss_batch, accuracy_batch = self._perform_batch_iteration(
    #             batch, targets, lengths, i, train_mode=False)
    #         accuracies.append(accuracy_batch)
    #         losses.append(loss_batch)

    #     return float(np.mean(losses)), float(np.mean(accuracies))
