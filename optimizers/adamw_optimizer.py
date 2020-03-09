from torch import optim
from torch.optim.optimizer import Optimizer

from models.model_base import ModelBase

from optimizers.optimizer_base import OptimizerBase
from services.arguments.arguments_service_base import ArgumentsServiceBase

from transformers import AdamW


class AdamWOptimizer(OptimizerBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            model: ModelBase):
        super(AdamWOptimizer, self).__init__()

        learning_rate = arguments_service.learning_rate
        self._optimizer = AdamW(model.parameters(), lr=learning_rate)

    def step(self):
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()