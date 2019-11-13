from torch import optim
from torch.optim.optimizer import Optimizer

from models.model_base import ModelBase

from services.arguments_service_base import ArgumentsServiceBase

class OptimizerService:
    def __init__(
        self,
        arguments_service: ArgumentsServiceBase,
        model: ModelBase):
        # TODO Add optimizer parameter

        self._optimizer = optim.Adam(model.parameters())

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer