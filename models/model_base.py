import os

import torch
import torch.nn as nn

from datetime import datetime

from entities.model_checkpoint import ModelCheckpoint

from services.data_service import DataService
from services.arguments_service_base import ArgumentsServiceBase


class ModelBase(nn.Module):
    def __init__(
            self,
            data_service: DataService,
            arguments_service: ArgumentsServiceBase):
        super(ModelBase, self).__init__()

        self._data_service = data_service
        self._arguments_service = arguments_service

    def forward(self):
        return None

    def calculate_accuracy(self, predictions, targets) -> bool:
        return 0

    def compare_metric(self, best_metric, metrics) -> bool:
        return True

    def clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    def save(
            self,
            path: str,
            epoch: int,
            iteration: int,
            best_metrics: object,
            name_prefix: str = None,
            save_model_dict: bool = True) -> bool:
        model_checkpoint = ModelCheckpoint(
            model_dict=self.state_dict() if save_model_dict else {},
            epoch=epoch,
            iteration=iteration,
            best_metrics=best_metrics)

        checkpoint_name = self._get_model_name(name_prefix)
        saved = self._data_service.save_python_obj(
            model_checkpoint, path, checkpoint_name, print_success=False)

        return saved

    def load(
            self,
            path: str,
            name_prefix: str = None,
            load_model_dict: bool = True) -> ModelCheckpoint:
        checkpoint_name = self._get_model_name(name_prefix)

        if not self._data_service.python_obj_exists(path, checkpoint_name):
            return None

        model_checkpoint: ModelCheckpoint = self._data_service.load_python_obj(
            path, checkpoint_name)

        if not model_checkpoint:
            raise Exception('Model checkpoint not found')

        if load_model_dict:
            self.load_state_dict(model_checkpoint.model_dict)

        return model_checkpoint

    def _get_model_name(self, name_prefix: str = None) -> str:
        checkpoint_name = self._arguments_service.get_argument(
            'checkpoint_name')
        result = f'checkpoint-{checkpoint_name}'

        if name_prefix:
            result = name_prefix + result

        return result
