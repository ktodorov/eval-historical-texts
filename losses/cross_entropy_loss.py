import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self._criterion = nn.CrossEntropyLoss(ignore_index=0)

    def backward(self, model_output):
        output, trg, lengths = model_output
        output_dim = output.shape[-1]

        sequences_length = output.shape[1] - 1
        batch_size = output.shape[0]

        output = output[:,1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        new_output = torch.zeros(lengths.sum(), output_dim).to('cuda')
        new_trg = torch.zeros(lengths.sum(), dtype=torch.long).to('cuda')
        counter = 0
        for i in range(batch_size):
            new_output[counter:counter+lengths[i]] = output[(i * sequences_length):((i * sequences_length) + lengths[i])]
            new_trg[counter:counter+lengths[i]] = trg[(i * sequences_length):((i * sequences_length) + lengths[i])]
            counter += lengths[i]

        output = new_output
        trg = new_trg

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = self._criterion.forward(output, trg)
        loss.backward()

        return loss.item()

    def calculate_loss(self, model_output):
        raise Exception('s')
        # loss = model_output
        # return loss.item()
