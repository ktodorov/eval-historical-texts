import torch
import torch.nn as nn


from services.arguments.arguments_service_base import ArgumentsServiceBase
from losses.sequence_loss import SequenceLoss

class TransformerSequenceLoss(SequenceLoss):
    def __init__(self):
        super().__init__()
        self._criterion = nn.CrossEntropyLoss(ignore_index=0)

    def _calculate_inner_loss(self, model_output):
        output, targets, lengths = model_output
        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        targets = targets[:,1:].contiguous().view(-1)

        loss = self._criterion.forward(output, targets)
        return loss