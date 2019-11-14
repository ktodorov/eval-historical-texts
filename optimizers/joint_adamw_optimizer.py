from torch import optim
from torch.optim.optimizer import Optimizer

from models.model_base import ModelBase

from optimizers.optimizer_base import OptimizerBase
from services.arguments_service_base import ArgumentsServiceBase

from transformers import AdamW


class JointAdamWOptimizer(OptimizerBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            model: ModelBase):
        super(JointAdamWOptimizer, self).__init__()

        model1_parameters, model2_parameters = model.parameters()
        learning_rate = arguments_service.get_argument('learning_rate')
        self._optimizer1 = AdamW(model1_parameters, lr=learning_rate)
        self._optimizer2 = AdamW(model2_parameters, lr=learning_rate)

    def step(self):
        self._optimizer1.step()
        self._optimizer2.step()

    def zero_grad(self):
        self._optimizer1.zero_grad()
        self._optimizer2.zero_grad()