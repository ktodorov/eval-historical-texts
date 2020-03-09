from torch.optim.optimizer import Optimizer

from models.model_base import ModelBase
from services.arguments.arguments_service_base import ArgumentsServiceBase

from transformers import AdamW


class OptimizerBase():
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            model: ModelBase):
        super(OptimizerBase, self).__init__()

        self._model = model
        self._learning_rate = arguments_service.learning_rate
        self._optimizer = None

    def get_optimizer(self) -> Optimizer:
        if self._optimizer is None:
            self._optimizer = self._init_optimizer()

        return self._optimizer

    def step(self):
        pass

    def zero_grad(self):
        pass

    def _init_optimizer(self) -> Optimizer:
        pass
