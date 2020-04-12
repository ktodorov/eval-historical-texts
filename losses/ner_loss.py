import torch
import torch.nn as nn


from services.arguments.arguments_service_base import ArgumentsServiceBase
from losses.loss_base import LossBase

class NERLoss(LossBase):
    def __init__(self):
        super().__init__()
        # self._criterion = nn.CrossEntropyLoss(ignore_index=0)

    def backward(self, model_output):
        loss = self._calculate_inner_loss(model_output)
        loss.backward()

        return loss.item()

    def calculate_loss(self, model_output):
        loss = self._calculate_inner_loss(model_output)
        return loss.item()

    def _calculate_inner_loss(self, model_output):
        loss, _, _ = model_output
        return loss

        # outputs, targets = model_output
        # #reshape labels to give a flat vector of length batch_size*seq_len
        # targets = targets.view(-1)  

        # #mask out 'PAD' tokens
        # mask = (targets >= 0).float()

        # #the number of tokens is the sum of elements in mask
        # num_tokens = int(torch.sum(mask).item())

        # #pick the values corresponding to targets and multiply by mask
        # outputs = outputs[range(outputs.shape[0]), targets]*mask

        # #cross entropy loss for all non 'PAD' tokens
        # return -torch.sum(outputs)/num_tokens

    # @property
    # def criterion(self) -> nn.Module:
    #     return self._criterion