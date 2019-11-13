import torch.nn as nn


class ModelBase(nn.Module):
    def __init__(self):
        super(ModelBase, self).__init__()

    def forward(self):
        return None

    def calculate_accuracy(self, predictions, targets):
        return 0
