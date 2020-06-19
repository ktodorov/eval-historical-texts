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
from entities.data_output_log import DataOutputLog

from enums.metric_type import MetricType

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService
from services.dataloader_service import DataLoaderService
from services.file_service import FileService
from services.log_service import LogService

from transformers import BertTokenizer


class TrainService:
    def __init__(
            self,
            arguments_service: PretrainedArgumentsService,
            dataloader_service: DataLoaderService,
            loss_function: LossBase,
            optimizer: OptimizerBase,
            log_service: LogService,
            file_service: FileService,
            model: ModelBase):

        self._arguments_service = arguments_service
        self._model_path = file_service.get_checkpoints_path()
        self._optimizer_base = optimizer

        self._log_service = log_service
        self._dataloader_service = dataloader_service

        self._loss_function = loss_function
        self._model = model.to(arguments_service.device)
        self.data_loader_train: DataLoader = None
        self.data_loader_validation: DataLoader = None

        self._initial_patience = self._arguments_service.patience
        # if we are going to fine-tune after initial convergence
        # then we set a low patience first and use the real one in
        # the second training iteration set
        if self._arguments_service.fine_tune_after_convergence:
            self._initial_patience = 5

    def train(self) -> bool:
        """
         main training function
        """
        epoch = 0

        try:
            self._log_service.initialize_evaluation()

            best_metrics = Metric(amount_limit=None)
            patience = self._initial_patience

            metric = Metric(amount_limit=self._arguments_service.eval_freq)

            start_epoch = 0
            start_iteration = 0
            resets_left = self._arguments_service.resets_limit
            reset_epoch_limit = self._arguments_service.training_reset_epoch_limit

            if self._arguments_service.resume_training:
                model_checkpoint = self._load_model()
                if model_checkpoint and not self._arguments_service.skip_best_metrics_on_resume:
                    best_metrics = model_checkpoint.best_metrics
                    start_epoch = model_checkpoint.epoch
                    start_iteration = model_checkpoint.iteration
                    resets_left = model_checkpoint.resets_left
                    metric.initialize(best_metrics)

            self.data_loader_train, self.data_loader_validation = self._dataloader_service.get_train_dataloaders()
            self._optimizer = self._optimizer_base.get_optimizer()
            self._log_service.start_logging_model(
                self._model, self._loss_function.criterion)

            # run
            epoch = start_epoch
            model_has_converged = False
            while epoch < self._arguments_service.epochs:
                self._log_service.log_summary('Epoch', epoch)

                best_metrics, patience = self._perform_epoch_iteration(
                    epoch, best_metrics, patience, metric, resets_left, start_iteration)

                start_iteration = 0  # reset the starting iteration

                # flush prints
                sys.stdout.flush()

                if patience == 0:
                    # we only prompt the model for changes on convergence once
                    should_start_again = not model_has_converged and self._model.on_convergence()
                    if should_start_again:
                        model_has_converged = True
                        model_checkpoint = self._load_model()
                        if model_checkpoint is not None:
                            best_metrics = model_checkpoint.best_metrics
                            start_epoch = model_checkpoint.epoch
                            start_iteration = model_checkpoint.iteration
                            resets_left = model_checkpoint.resets_left
                            metric.initialize(best_metrics)

                        self._initial_patience = self._arguments_service.patience
                        patience = self._initial_patience
                        epoch += 1
                    elif (self._arguments_service.reset_training_on_early_stop and resets_left > 0 and reset_epoch_limit > epoch):
                        patience = self._initial_patience
                        resets_left -= 1
                        self._log_service.log_summary(
                            key='Resets left', value=resets_left)

                        print(
                            f'Resetting training due to early stop activated. Resets left: {resets_left}')
                    else:
                        print('Stopping training due to depleted patience')
                        break
                else:
                    epoch += 1

            if epoch >= self._arguments_service.epochs:
                print('Stopping training due to depleted epochs')

        except KeyboardInterrupt as e:
            print(f"Killed by user: {e}")
            if self._arguments_service.save_checkpoint_on_crash:
                self._model.save(
                    self._model_path,
                    epoch,
                    0,
                    best_metrics,
                    resets_left,
                    name_prefix=f'KILLED_at_epoch_{epoch}')

            return False
        except Exception as e:
            print(e)
            if self._arguments_service.save_checkpoint_on_crash:
                self._model.save(
                    self._model_path,
                    epoch,
                    0,
                    best_metrics,
                    resets_left,
                    name_prefix=f'CRASH_at_epoch_{epoch}')
            raise e

        # flush prints
        sys.stdout.flush()

        if self._arguments_service.save_checkpoint_on_finish:
            self._model.save(
                self._model_path,
                epoch,
                0,
                best_metrics,
                resets_left,
                name_prefix=f'FINISHED_at_epoch_{epoch}')

        return True

    def _perform_epoch_iteration(
            self,
            epoch_num: int,
            best_metrics: Metric,
            patience: int,
            metric: Metric,
            resets_left: int,
            start_iteration: int = 0) -> Tuple[Metric, int]:
        """
        one epoch implementation
        """
        data_loader_length = len(self.data_loader_train)

        for i, batch in enumerate(self.data_loader_train):
            if i < start_iteration:
                continue

            self._log_service.log_progress(i, data_loader_length, epoch_num)

            loss_batch, accuracies_batch, _ = self._perform_batch_iteration(
                batch)
            assert not math.isnan(
                loss_batch), f'loss is NaN during training at iteration {i}'

            metric.add_loss(loss_batch)
            metric.add_accuracies(accuracies_batch)

            # calculate amount of batches and walltime passed
            batches_passed = i + (epoch_num * data_loader_length)

            # run on validation set and print progress to terminal
            # if we have eval_frequency or if we have finished the epoch
            if self._should_evaluate(batches_passed, i, data_loader_length):
                if not self._arguments_service.skip_validation:
                    validation_metric = self._evaluate()
                else:
                    validation_metric = Metric(metric=metric)

                assert not math.isnan(metric.get_current_loss(
                )), f'combined loss is NaN during training at iteration {i}; losses are - {metric._losses}'

                new_best = self._model.compare_metric(
                    best_metrics, validation_metric)
                if new_best:
                    best_metrics, patience = self._save_current_best_result(
                        validation_metric, epoch_num, i, resets_left)
                else:
                    patience -= 1

                self._log_service.log_evaluation(
                    metric,
                    validation_metric,
                    batches_passed,
                    epoch_num,
                    i,
                    data_loader_length,
                    new_best,
                    metric_log_key=self._model.metric_log_key)

                self._log_service.log_summary(
                    key='Patience left', value=patience)

                self._model.finalize_batch_evaluation(is_new_best=new_best)

            # check if runtime is expired
            self._validate_time_passed()

            if patience == 0:
                break

        return best_metrics, patience

    def _perform_batch_iteration(
            self,
            batch: torch.Tensor,
            train_mode: bool = True,
            output_characters: bool = False) -> Tuple[float, Dict[MetricType, float]]:
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

        metrics, current_output_log = self._model.calculate_accuracies(
            batch, outputs, output_characters=output_characters)

        return loss, metrics, current_output_log

    def _load_model(self) -> ModelCheckpoint:
        model_checkpoint = self._model.load(self._model_path, 'BEST')
        if not model_checkpoint:
            model_checkpoint = self._model.load(self._model_path)

        return model_checkpoint

    def _evaluate(self) -> Metric:
        metric = Metric(amount_limit=None)
        data_loader_length = len(self.data_loader_validation)
        full_output_log = DataOutputLog()

        for i, batch in enumerate(self.data_loader_validation):
            if not batch:
                continue

            self._log_service.log_progress(
                i, data_loader_length, evaluation=True)

            loss_batch, metrics_batch, current_output_log = self._perform_batch_iteration(
                batch, train_mode=False, output_characters=(len(full_output_log) < 100))

            if math.isnan(loss_batch):
                raise Exception(
                    f'loss is NaN during evaluation at iteration {i}')

            if current_output_log is not None:
                full_output_log.extend(current_output_log)

            metric.add_accuracies(metrics_batch)
            metric.add_loss(loss_batch)

        final_metric = self._model.calculate_evaluation_metrics()
        metric.add_accuracies(final_metric)
        self._log_service.log_batch_results(full_output_log)

        assert not math.isnan(metric.get_current_loss(
        )), f'combined loss is NaN during evaluation at iteration {i}; losses are - {metric._losses}'

        return metric

    def _should_evaluate(
            self,
            batches_passed: int,
            iteration: int,
            data_loader_length: int):
        # If we don't use validation set, then we must not evaluate before we pass at least `eval_freq` batches
        if self._arguments_service.skip_validation and batches_passed < self._arguments_service.eval_freq:
            return False

        result = (batches_passed % self._arguments_service.eval_freq) == 0
        return result

    def _validate_time_passed(self):
        time_passed = self._log_service.get_time_passed()
        if ((time_passed.total_seconds() > (self._arguments_service.max_training_minutes * 60)) and
                self._arguments_service.max_training_minutes > 0):
            raise KeyboardInterrupt(
                f"Process killed because {self._arguments_service.max_training_minutes} minutes passed")

    def _save_current_best_result(
            self,
            validation_metric: Metric,
            epoch_num: int,
            i: int,
            resets_left: int):
        best_metrics = validation_metric
        self._model.save(self._model_path, epoch_num, i,
                         best_metrics, resets_left, name_prefix=f'BEST')

        best_accuracies = best_metrics.get_current_accuracies()

        for key, value in best_accuracies.items():
            self._log_service.log_summary(
                key=f'Best - {str(key)}', value=value)

        self._log_service.log_summary(
            key='Best loss', value=best_metrics.get_current_loss())
        patience = self._initial_patience

        return best_metrics, patience
