import torch
import torch.nn as nn
from overrides import overrides


from losses.loss_base import LossBase

from services.arguments.arguments_service_base import ArgumentsServiceBase

class SequenceLoss(LossBase):
    def __init__(self):
        super().__init__()

        self._criterion = nn.NLLLoss(reduction="sum", ignore_index=0)

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
        outputs, targets, _ = model_output
        batch_size = targets.shape[0]

        target_y = targets[:, 1:]

        loss = self._criterion.forward(
            outputs.contiguous().view(-1, outputs.size(-1)),
            target_y.contiguous().view(-1))

        loss = loss / batch_size
        return loss

    @property
    @overrides
    def criterion(self) -> nn.Module:
        return self._criterion