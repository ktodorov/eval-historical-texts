import os
import numpy as np
import torch
from typing import List

from entities.model_checkpoint import ModelCheckpoint

from models.model_base import ModelBase

from services.semantic_arguments_service import SemanticArgumentsService
from services.data_service import DataService
from services.model_service import ModelService


class JointModel(ModelBase):
    def __init__(
            self,
            arguments_service: SemanticArgumentsService,
            data_service: DataService,
            model_service: ModelService):
        super(JointModel, self).__init__(data_service)

        self._number_of_models: int = arguments_service.joint_model_amount

        self._inner_models: List[ModelBase] = [
            model_service.create_model() for _ in range(self._number_of_models)]

    def forward(self, inputs, **kwargs):

        result = []
        for i, model in enumerate(self._inner_models):
            current_input = inputs[i] if len(inputs) == len(self._inner_models) else inputs
            outputs = model.forward(current_input)
            result.append(outputs)

        return result

    def named_parameters(self):
        return [model.named_parameters() for model in self._inner_models]

    def parameters(self):
        return [model.parameters() for model in self._inner_models]

    def calculate_accuracy(self, predictions, targets) -> int:
        return 0

    def compare_metric(self, best_metric, metrics) -> bool:
        if best_metric is None or np.sum(best_metric) > np.sum(metrics):
            return True

    def clip_gradients(self):
        model_parameters = self.parameters()

        [torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
         for parameters in model_parameters]

    def save(
            self,
            path: str,
            epoch: int,
            iteration: int,
            best_metrics: object,
            resets_left: int,
            name_prefix: str = None) -> bool:

        saved = super().save(path, epoch, iteration, best_metrics,
                             resets_left, name_prefix, save_model_dict=False)

        if not saved:
            return saved

        for i, model in enumerate(self._inner_models):
            model.save(path, epoch, iteration,
                       best_metrics, f'{name_prefix}_{i}')

        return saved

    def load(self, path: str, name_prefix: str = None) -> ModelCheckpoint:
        for i, model in enumerate(self._inner_models):
            model.load(
                path,
                f'{name_prefix}_{i+1}',
                load_model_only=True)

        return None
