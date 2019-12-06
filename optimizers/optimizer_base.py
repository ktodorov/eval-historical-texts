from torch.optim.optimizer import Optimizer

from models.model_base import ModelBase
from services.arguments_service_base import ArgumentsServiceBase

from transformers import AdamW


class OptimizerBase():
    def __init__(self):
        super(OptimizerBase, self).__init__()

    def step(self):
        pass

    def zero_grad(self):
        pass