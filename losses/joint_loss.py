import torch.nn as nn

class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()

    def backward(self, models_outputs):
        for model_output in models_outputs:
            model_output.backward()

        result = [model_output.item() for model_output in models_outputs]
        return result

    def calculate_loss(self, models_outputs):
        result = [model_output.item() for model_output in models_outputs]
        return result
