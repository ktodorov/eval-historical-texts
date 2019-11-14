import numpy as np
import torch
from transformers import BertForMaskedLM

from models.model_base import ModelBase

from services.arguments_service_base import ArgumentsServiceBase


class JointKBertModel(ModelBase):
    def __init__(self, arguments_service: ArgumentsServiceBase):
        super(JointKBertModel, self).__init__()

        self._bert_model1 = BertForMaskedLM.from_pretrained(
            arguments_service.get_argument('pretrained_weights'))

        self._bert_model2 = BertForMaskedLM.from_pretrained(
            arguments_service.get_argument('pretrained_weights'))

    def forward(self, input_batch, **kwargs):
        ((inputs1, labels1), (inputs2, labels2)) = input_batch
        outputs1 = self._bert_model1.forward(inputs1, masked_lm_labels=labels1)
        outputs2 = self._bert_model2.forward(inputs2, masked_lm_labels=labels2)
        return (outputs1, outputs2)

    def named_parameters(self):
        return (self._bert_model1.named_parameters(), self._bert_model2.named_parameters())

    def parameters(self):
        return (self._bert_model1.parameters(), self._bert_model2.parameters())

    def calculate_accuracy(self, predictions, targets) -> int:
        return 0

    def compare_metric(self, best_metric, metrics) -> bool:
        if best_metric is None or np.sum(best_metric) > np.sum(metrics):
            return True

    def clip_gradients(self):
        (parameters1, parameters2) = self.parameters()
        torch.nn.utils.clip_grad_norm_(parameters1, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(parameters2, max_norm=1.0)
