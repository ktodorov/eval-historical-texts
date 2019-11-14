import torch
import torch.nn as nn


class ModelBase(nn.Module):
    def __init__(self):
        super(ModelBase, self).__init__()

    def forward(self):
        return None

    def calculate_accuracy(self, predictions, targets) -> bool:
        return 0

    def compare_metric(self, best_metric, metrics) -> bool:
        return True

    def clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)