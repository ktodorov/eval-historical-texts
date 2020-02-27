import torch
import torch.nn as nn


from losses.loss_base import LossBase

class SequenceLoss(LossBase):
    def __init__(self):
        super().__init__()
        self._criterion = nn.CrossEntropyLoss(ignore_index=0)

    def backward(self, model_output):
        loss = self._calculate_inner_loss(model_output)
        loss.backward()

        return loss.item()

    def calculate_loss(self, model_output):
        loss = self._calculate_inner_loss(model_output)
        return loss.item()

    def _calculate_inner_loss(self, model_output):
        output, targets, lengths = model_output
        output_dim = output.shape[-1]

        sequences_length = output.shape[1]
        batch_size = output.shape[0]

        output = output.reshape(-1, output_dim)
        targets = targets.reshape(-1)

        loss = self._criterion.forward(output, targets)
        return loss

    @property
    def criterion(self) -> nn.Module:
        return self._criterion