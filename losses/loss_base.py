import torch.nn as nn

class LossBase(nn.Module):
    def __init__(self):
        super(LossBase, self).__init__()
