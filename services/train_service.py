import os
import sys
import time
import math
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from losses.loss_base import LossBase
from models.model_base import ModelBase
from optimizers.optimizer_base import OptimizerBase

from entities.model_checkpoint import ModelCheckpoint
from entities.metric import Metric

from enums.metric_type import MetricType

from services.arguments_service_base import ArgumentsServiceBase
from services.dataloader_service import DataLoaderService
from services.file_service import FileService
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
            file_service: FileService,
            model: ModelBase):

        self._arguments_service = arguments_service
        self._model_path = file_service.get_checkpoints_path()

        self._log_service = log_service
        self._dataloader_service = dataloader_service

        self._loss_function = loss_function
        self._optimizer = optimizer
        self._model = model.to(arguments_service.get_argument('device'))
        self._patience: int = self._arguments_service.get_argument('patience')
        self._debug: bool = self._arguments_service.get_argument('debug')

    def train(self) -> bool:
        """
         main training function
        """

        (self.data_loader_train,
         self.data_loader_validation) = self._dataloader_service.get_train_dataloaders()

        epoch = 0
        self._log_service.start_logging_model(
            self._model, self._loss_function.criterion)

        try:
            self._log_service.initialize_evaluation()

            best_metrics = Metric(amount_limit=None)
            patience = self._patience
            metric = Metric(amount_limit=10)

            start_epoch = 0
            start_iteration = 0

            if self._arguments_service.get_argument('resume_training'):
                model_checkpoint = self._load_model()
                if model_checkpoint:
                    best_metrics = model_checkpoint.best_metrics
                    start_epoch = model_checkpoint.epoch
                    start_iteration = model_checkpoint.iteration
                    metric.initialize(best_metrics, start_iteration)

            # run
            for epoch in range(start_epoch, self._arguments_service.get_argument('epochs')):
                self._log_service.log_summary('Epoch', epoch)

                best_metrics, patience = self._perform_epoch_iteration(
                    epoch, best_metrics, patience, metric, start_iteration)

                start_iteration = 0  # reset the starting iteration

                # flush prints
                sys.stdout.flush()

                if patience == 0:
                    print('Stopping training due to depleted patience')
                    break

        except KeyboardInterrupt as e:
            print(f"Killed by user: {e}")
            self._model.save(self._model_path, epoch, 0, best_metrics,
                             name_prefix=f'KILLED_at_epoch_{epoch}')
            return False
        except Exception as e:
            print(e)
            self._model.save(self._model_path, epoch, 0, best_metrics,
                             name_prefix=f'CRASH_at_epoch_{epoch}')
            raise e

        # flush prints
        sys.stdout.flush()

        # example last save
        self._model.save(self._model_path, epoch, 0, best_metrics,
                         name_prefix=f'FINISHED_at_epoch_{epoch}')
        return True

    def _perform_epoch_iteration(
            self,
            epoch_num: int,
            best_metrics: Metric,
            patience: int,
            metric: Metric,
            start_iteration: int = 0) -> Tuple[Metric, int]:
        """
        one epoch implementation
        """
        train_accuracy = 0.0
        train_loss = 0.0
        data_loader_length = len(self.data_loader_train)

        for i, batch in enumerate(self.data_loader_train):
            if i < start_iteration:
                continue

            self._log_service.log_progress(i, data_loader_length)

            loss_batch, accuracies_batch = self._perform_batch_iteration(
                batch, debug=self._debug)
            if math.isnan(loss_batch):
                raise Exception(
                    f'loss is NaN during training at iteration {i}')

            metric.add_loss(loss_batch)
            metric.add_accuracies(accuracies_batch)

            # calculate amount of batches and walltime passed
            batches_passed = i + (epoch_num * len(self.data_loader_train))

            # run on validation set and print progress to terminal
            # if we have eval_frequency or if we have finished the epoch
            if (batches_passed % self._arguments_service.get_argument('eval_freq')) == 0 or (i + 1 == data_loader_length):
                validation_metric = self._evaluate()

                if math.isnan(validation_metric.get_current_loss()):
                    raise Exception(
                        f'combined loss is NaN during evaluating at iteration {i}; losses are - {validation_metric._losses}')

                if math.isnan(metric.get_current_loss()):
                    raise Exception(
                        f'combined loss is NaN during evaluating at iteration {i}; losses are - {metric._losses}')

                new_best = self._model.compare_metric(
                    best_metrics, validation_metric)
                if new_best:
                    best_metrics = validation_metric
                    self._model.save(self._model_path, epoch_num, i,
                                     best_metrics, name_prefix=f'BEST')

                    best_accuracies = best_metrics.get_current_accuracies()
                    for key, value in best_accuracies.items():
                        self._log_service.log_summary(
                            key=f'Best - {str(key)}', value=value)
                    self._log_service.log_summary(
                        key='Best loss', value=best_metrics.get_current_loss())
                    patience = self._patience
                else:
                    patience -= 1

                self._log_service.log_evaluation(
                    metric,
                    validation_metric,
                    batches_passed,
                    epoch_num,
                    i,
                    data_loader_length,
                    new_best)

                self._log_service.log_summary(
                    key='Patience left', value=patience)

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
            train_mode: bool = True,
            print_characters: bool = False,
            debug: bool = False) -> Tuple[float, Dict[MetricType, float]]:
        """
        runs forward pass on batch and backward pass if in train_mode
        """

        if train_mode:
            self._model.train()
            self._optimizer.zero_grad()
        else:
            self._model.eval()

        outputs = self._model.forward(batch, debug=debug)

        accuracy = 0
        if train_mode:
            loss = self._loss_function.backward(outputs)
            self._model.clip_gradients()
            self._optimizer.step()
            self._model.zero_grad()
        else:
            loss = self._loss_function.calculate_loss(outputs)

        accuracies = self._model.calculate_accuracies(
            batch, outputs, print_characters=print_characters)

        return loss, accuracies

    def _load_model(self) -> ModelCheckpoint:
        model_checkpoint = self._model.load(self._model_path, 'BEST')
        if not model_checkpoint:
            model_checkpoint = self._model.load(self._model_path)

        return model_checkpoint

    def _evaluate(self) -> Metric:
        metric = Metric(amount_limit=None)
        data_loader_length = len(self.data_loader_validation)

        for i, batch in enumerate(self.data_loader_validation):
            if not batch:
                continue

            self._log_service.log_progress(
                i, data_loader_length, evaluation=True)

            loss_batch, accuracies_batch = self._perform_batch_iteration(
                batch, train_mode=False, print_characters=True)

            if math.isnan(loss_batch):
                raise Exception(
                    f'loss is NaN during evaluation at iteration {i}')

            metric.add_accuracies(accuracies_batch)
            metric.add_loss(loss_batch)

        return metric
