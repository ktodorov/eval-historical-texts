import os

import torch
import torch.nn as nn

from datetime import datetime

from typing import Dict, List, Tuple

from overrides import overrides

from entities.model_checkpoint import ModelCheckpoint
from entities.metric import Metric
from entities.batch_representation import BatchRepresentation
from enums.metric_type import MetricType

from services.data_service import DataService
from services.arguments.arguments_service_base import ArgumentsServiceBase

class ModelBase(nn.Module):
    def __init__(
            self,
            data_service: DataService = None,
            arguments_service: ArgumentsServiceBase = None):
        super(ModelBase, self).__init__()

        self._data_service = data_service
        self._arguments_service = arguments_service
        self.do_not_save: bool = False

        self.metric_log_key: str = None

    def forward(self, batch_representation: BatchRepresentation):
        return None

    def calculate_accuracies(
        self,
        batch: BatchRepresentation,
        outputs,
        output_characters=False) -> Tuple[Dict[MetricType, float], List[str]]:
        return ({}, None)

    def compare_metric(self, best_metric: Metric, new_metric: Metric) -> bool:
        if best_metric.is_new:
            return True

        current_best = 0
        new_result = 0

        if self.metric_log_key is not None:
            current_best = best_metric.get_accuracy_metric(self.metric_log_key)
            new_result = new_metric.get_accuracy_metric(self.metric_log_key)

        if current_best == new_result:
            result = best_metric.get_current_loss() >= new_metric.get_current_loss()
        else:
            result = current_best < new_result

        return result

    def clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    def save(
            self,
            path: str,
            epoch: int,
            iteration: int,
            best_metrics: object,
            resets_left: int,
            name_prefix: str = None,
            save_model_dict: bool = True) -> bool:
        assert self._data_service is not None
        assert self._arguments_service is not None

        model_checkpoint = ModelCheckpoint(
            model_dict=self.state_dict() if save_model_dict else {},
            epoch=epoch,
            iteration=iteration,
            best_metrics=best_metrics,
            resets_left=resets_left)

        checkpoint_name = self._get_model_name(name_prefix)
        saved = self._data_service.save_python_obj(
            model_checkpoint, path, checkpoint_name, print_success=False)

        return saved

    def load(
            self,
            path: str,
            name_prefix: str = None,
            load_model_dict: bool = True,
            load_model_only: bool = False,
            use_checkpoint_name: bool = True,
            checkpoint_name: str = None) -> ModelCheckpoint:
        assert self._data_service is not None
        assert self._arguments_service is not None

        if checkpoint_name is None:
            if not use_checkpoint_name:
                checkpoint_name = name_prefix
            else:
                checkpoint_name = self._arguments_service.resume_checkpoint_name
                if checkpoint_name is None:
                    checkpoint_name = self._get_model_name(name_prefix)

        if load_model_only:
            return None

        if not self._data_service.python_obj_exists(path, checkpoint_name):
            raise Exception(f'Model checkpoint "{checkpoint_name}" not found at "{path}"')

        model_checkpoint: ModelCheckpoint = self._data_service.load_python_obj(
            path, checkpoint_name)

        if model_checkpoint is None:
            raise Exception('Model checkpoint not found')

        if load_model_dict:
            ignored_parameters = []
            model_dict = model_checkpoint.model_dict

            for module_name, module in self.named_modules():
                if isinstance(module, ModelBase):
                    if module.do_not_save:
                        for parameter_name, parameter_value in module.named_parameters():
                            model_dict[f'{module_name}.{parameter_name}'] = parameter_value

                    module.before_load()

            self.load_state_dict(model_dict)

            for module_name, module in self.named_modules():
                if isinstance(module, ModelBase):
                    module.after_load()

        return model_checkpoint

    def _get_model_name(self, name_prefix: str = None) -> str:
        result = 'checkpoint'
        if self._arguments_service.checkpoint_name is not None:
            result = self._arguments_service.checkpoint_name

        if name_prefix:
            result = f'{name_prefix}_{result}'

        return result

    def on_convergence(self) -> bool:
        result = self._on_convergence(self)
        for _, module in self.named_modules():
            result = result or self._on_convergence(module)

        return result

    def _on_convergence(self, main_module) -> bool:
        result = False
        for module_name, module in main_module.named_modules():
            if module_name == '':
                continue

            if isinstance(module, ModelBase):
                result = result or module.on_convergence()

        return result

    @overrides
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if self.do_not_save:
            return None

        result = super().state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars)

        return result

    @property
    def keep_frozen(self) -> bool:
        return False

    @overrides
    def train(self, mode=True):
        # If fine-tuning is disabled, we don't set the module to train mode
        if mode and self.keep_frozen:
            self.requires_grad_(requires_grad=False)

            return

        if mode:
            self.requires_grad_(requires_grad=True)

        super().train(mode)

    @overrides
    def eval(self):
        super().eval()

        self.requires_grad_(requires_grad=False)

    def optimizer_parameters(self):
        return self.parameters()


    def calculate_evaluation_metrics(self) -> Dict[str, float]:
        return {}

    def finalize_batch_evaluation(self, is_new_best: bool):
        pass

    def before_load(self):
        pass

    def after_load(self):
        pass