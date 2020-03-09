from torch import optim
from torch.optim.optimizer import Optimizer

from models.model_base import ModelBase

from optimizers.optimizer_base import OptimizerBase
from services.arguments.arguments_service_base import ArgumentsServiceBase

from transformers import AdamW


class JointAdamWOptimizer(OptimizerBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            model: ModelBase):
        super().__init__(arguments_service, model)

    def _init_optimizer(self) -> Optimizer:
        model1_parameters, model2_parameters = self._model.parameters()
        optimizer1 = AdamW(model1_parameters, lr=self._learning_rate)
        optimizer2 = AdamW(model2_parameters, lr=self._learning_rate)
        return (optimizer1, optimizer2)

    def step(self):
        self._optimizer[0].step()
        self._optimizer[1].step()

    def zero_grad(self):
        self._optimizer[0].zero_grad()
        self._optimizer[1].zero_grad()