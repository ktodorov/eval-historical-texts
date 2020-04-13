import torch.nn as nn
from overrides import overrides

from losses.loss_base import LossBase

class JointLoss(LossBase):
    def __init__(self):
        super(JointLoss, self).__init__()

    @overrides
    def backward(self, models_outputs):
        for model_output in models_outputs:
            model_output.backward()

        result = [model_output.item() for model_output in models_outputs]
        return result

    @overrides
    def calculate_loss(self, models_outputs):
        result = [model_output.item() for model_output in models_outputs]
        return result
