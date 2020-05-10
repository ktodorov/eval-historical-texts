from torch import optim
from torch.optim.optimizer import Optimizer
from overrides import overrides

from models.model_base import ModelBase

from optimizers.optimizer_base import OptimizerBase
from services.arguments.arguments_service_base import ArgumentsServiceBase


class SGDOptimizer(OptimizerBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            model: ModelBase):
        super().__init__(arguments_service, model)

        self._momentum = arguments_service.momentum
        self._weight_decay = arguments_service.weight_decay

    def _init_optimizer(self) -> Optimizer:
        optimizer = optim.SGD(
            self._model.optimizer_parameters(),
            momentum=self._momentum,
            weight_decay=self._weight_decay,
            lr=self._learning_rate)

        return optimizer

    @overrides
    def step(self):
        self._optimizer.step()

    @overrides
    def zero_grad(self):
        self._optimizer.zero_grad()
