import torch.nn as nn

class KBertLoss(nn.Module):
    def __init__(self):
        super(KBertLoss, self).__init__()

    def backward(self, model_output):
        model_output.backward()

        return model_output.item()

    def calculate_loss(self, model_output):
        loss = model_output
        return loss.item()
