import torch.nn as nn

class KBertLoss(nn.Module):
    def __init__(self):
        super(KBertLoss, self).__init__()

    def backward(self, model_output):
        loss = model_output[0]
        loss.backward()

        return loss.item()

    def calculate_loss(self, model_output):
        loss = model_output[0]
        return loss.item()
