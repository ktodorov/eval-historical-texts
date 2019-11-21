import os
import numpy as np
import torch
from typing import List

from transformers import BertForMaskedLM

from entities.model_checkpoint import ModelCheckpoint

from models.model_base import ModelBase
from models.kbert_model import KBertModel

from services.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService


class JointKBertModel(ModelBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            data_service: DataService,
            number_of_models: int = 2):
        super(JointKBertModel, self).__init__(data_service, arguments_service)

        self._number_of_models = 2
        self._bert_models: List[ModelBase] = [KBertModel(
            arguments_service, data_service) for _ in range(number_of_models)]

    def forward(self, input_word, **kwargs):

        result = []
        for i, model in enumerate(self._bert_models):
            outputs = model.forward(input_word)
            result.append(outputs)

        return result

    def named_parameters(self):
        return [model.named_parameters() for model in self._bert_models]

    def parameters(self):
        return [model.parameters() for model in self._bert_models]

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
            name_prefix: str = None) -> bool:

        saved = super().save(path, epoch, iteration, best_metrics,
                             name_prefix, save_model_dict=False)

        if not saved:
            return saved

        for i, model in enumerate(self._bert_models):
            model.save(path, epoch, iteration, best_metrics, f'{name_prefix}_{i}_')

        return saved

    def load(self, path: str, name_prefix: str = None) -> ModelCheckpoint:

        # model_checkpoint = super().load(path, name_prefix, load_model_dict=False)
        # if not model_checkpoint:
        #     return None

        for i, model in enumerate(self._bert_models):
            model._load_kbert_model(path, f'{name_prefix}_{i+1}_')

        return None#model_checkpoint