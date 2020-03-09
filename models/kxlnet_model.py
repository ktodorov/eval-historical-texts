from transformers import XLNetLMHeadModel, BertPreTrainedModel

from models.kbert_model import KBertModel

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService

class KXLNetModel(KBertModel):
    def __init__(
        self,
        arguments_service: ArgumentsServiceBase,
        data_service: DataService):
        super(KXLNetModel, self).__init__(arguments_service, data_service)


    def forward(self, input_batch, **kwargs):
        if isinstance(input_batch, tuple):
            (inputs, labels) = input_batch
            outputs = self._bert_model.forward(inputs, labels=labels)
        else:
            inputs = input_batch
            outputs = self._bert_model.forward(inputs)

        return outputs[0]

    @property
    def _model_type(self) -> BertPreTrainedModel:
        return XLNetLMHeadModel