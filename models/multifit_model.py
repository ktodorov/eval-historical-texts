from torch import nn

from entities.model_checkpoint import ModelCheckpoint
from models.model_base import ModelBase

from services.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService

class MultiFitModel(ModelBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            data_service: DataService):
        super(MultiFitModel, self).__init__(data_service, arguments_service)

        # self._model = nn.LSTM()

    def forward(self, input_batch, **kwargs):
        pass
        # if isinstance(input_batch, tuple):
        #     (inputs, labels) = input_batch
        #     outputs = self._bert_model.forward(inputs, masked_lm_labels=labels)
        # else:
        #     inputs = input_batch
        #     outputs = self._bert_model.forward(inputs)

        # return outputs[0]

    def calculate_accuracy(self, predictions, targets) -> int:
        return 0

    def compare_metric(self, best_metric, metrics) -> bool:
        if best_metric is None or best_metric > metrics:
            return True