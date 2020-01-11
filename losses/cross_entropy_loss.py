import torch
import torch.nn as nn


from services.arguments_service_base import ArgumentsServiceBase
from losses.loss_base import LossBase

class CrossEntropyLoss(LossBase):
    def __init__(
            self,
            device: torch.device):
        super(CrossEntropyLoss, self).__init__()
        self._criterion = nn.CrossEntropyLoss(ignore_index=0)
        self._device = device

    def backward(self, model_output):
        loss = self._calculate_inner_loss(model_output)
        loss.backward()

        return loss.item()

    def calculate_loss(self, model_output):
        loss = self._calculate_inner_loss(model_output)
        return loss.item()

    def _calculate_inner_loss(self, model_output):
        output, targets, lengths = model_output
        output_dim = output.shape[-1]

        sequences_length = output.shape[1] - 1
        batch_size = output.shape[0]

        output = output[:, 1:].reshape(-1, output_dim)
        targets = targets[:, 1:].reshape(-1)

        new_output = torch.zeros(lengths.sum(), output_dim, device=self._device)
        new_trg = torch.zeros(lengths.sum(), dtype=torch.long, device=self._device)
        counter = 0
        for i in range(batch_size):
            new_output[counter:counter+lengths[i]
                       ] = output[(i * sequences_length):((i * sequences_length) + lengths[i])]
            new_trg[counter:counter+lengths[i]
                    ] = targets[(i * sequences_length):((i * sequences_length) + lengths[i])]
            counter += lengths[i]

        output = new_output
        targets = new_trg

        loss = self._criterion.forward(output, targets)
        return loss

    @property
    def criterion(self) -> nn.Module:
        return self._criterion