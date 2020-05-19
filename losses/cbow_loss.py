import torch
import torch.nn as nn

from overrides import overrides

from services.arguments.arguments_service_base import ArgumentsServiceBase
from losses.loss_base import LossBase

class CBOWLoss(LossBase):
    def __init__(self):
        super().__init__()

        self._criterion = nn.CrossEntropyLoss(ignore_index=0)

    @overrides
    def backward(self, model_output):
        loss = self._calculate_inner_loss(model_output)
        loss.backward()

        return loss.item()

    @overrides
    def calculate_loss(self, model_output):
        loss = self._calculate_inner_loss(model_output)
        return loss.item()

    def _calculate_inner_loss(self, model_output):
        log_probabilities, targets = model_output
        return self._criterion.forward(log_probabilities, targets)