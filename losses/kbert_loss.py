import torch.nn as nn
from overrides import overrides

from losses.loss_base import LossBase

class KBertLoss(LossBase):
    def __init__(self):
        super(KBertLoss, self).__init__()

    @overrides
    def backward(self, model_output):
        model_output.backward()

        return model_output.item()

    @overrides
    def calculate_loss(self, model_output):
        loss = model_output
        return loss.item()
