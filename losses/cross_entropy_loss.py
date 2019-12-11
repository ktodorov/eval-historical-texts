import torch
import torch.nn as nn


from services.arguments_service_base import ArgumentsServiceBase


class CrossEntropyLoss(nn.Module):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase):
        super(CrossEntropyLoss, self).__init__()
        self._criterion = nn.CrossEntropyLoss(ignore_index=0)
        self._arguments_service = arguments_service

    def backward(self, model_output):
        loss = self._calculate_inner_loss(model_output)
        loss.backward()

        return loss.item()

    def calculate_loss(self, model_output):
        loss = self._calculate_inner_loss(model_output)
        return loss.item()

    def _calculate_inner_loss(self, model_output):
        output, trg, lengths = model_output
        output_dim = output.shape[-1]

        sequences_length = output.shape[1] - 1
        batch_size = output.shape[0]

        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        new_output = torch.zeros(lengths.sum(), output_dim).to(self._arguments_service.get_argument('device'))
        new_trg = torch.zeros(lengths.sum(), dtype=torch.long).to(self._arguments_service.get_argument('device'))
        counter = 0
        for i in range(batch_size):
            new_output[counter:counter+lengths[i]
                       ] = output[(i * sequences_length):((i * sequences_length) + lengths[i])]
            new_trg[counter:counter+lengths[i]
                    ] = trg[(i * sequences_length):((i * sequences_length) + lengths[i])]
            counter += lengths[i]

        output = new_output
        trg = new_trg

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = self._criterion.forward(output, trg)
        return loss
