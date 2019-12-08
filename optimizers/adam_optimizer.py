from torch import optim
from torch.optim.optimizer import Optimizer

from models.model_base import ModelBase

from optimizers.optimizer_base import OptimizerBase
from services.arguments_service_base import ArgumentsServiceBase


class AdamOptimizer(OptimizerBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            model: ModelBase):
        super(AdamOptimizer, self).__init__()

        learning_rate = arguments_service.get_argument('learning_rate')
        self._optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def step(self):
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()