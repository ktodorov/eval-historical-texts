from torch import optim
from torch.optim.optimizer import Optimizer

from models.model_base import ModelBase

from optimizers.optimizer_base import OptimizerBase
from services.arguments.arguments_service_base import ArgumentsServiceBase


class AdamOptimizer(OptimizerBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            model: ModelBase):
        super().__init__(arguments_service, model)

    def _init_optimizer(self) -> Optimizer:
        optimizer = optim.Adam(
            self._model.parameters(),
            lr=self._learning_rate)

        return optimizer

    def step(self):
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()
