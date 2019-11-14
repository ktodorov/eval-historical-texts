from transformers import BertForMaskedLM

from models.model_base import ModelBase

from services.arguments_service_base import ArgumentsServiceBase


class KBertModel(ModelBase):
    def __init__(self, arguments_service: ArgumentsServiceBase):
        super(KBertModel, self).__init__()

        self._bert_model = BertForMaskedLM.from_pretrained(
            arguments_service.get_argument('pretrained_weights'))

    def forward(self, input_batch, **kwargs):
        (inputs, labels) = input_batch
        outputs = self._bert_model.forward(inputs, masked_lm_labels=labels)
        return outputs

    def named_parameters(self):
        return self._bert_model.named_parameters()

    def parameters(self):
        return self._bert_model.parameters()

    def calculate_accuracy(self, predictions, targets) -> int:
        return 0

    def compare_metric(self, best_metric, metrics) -> bool:
        if best_metric is None or best_metric > metrics:
            return True