import torch.nn as nn

class JointKBertLoss(nn.Module):
    def __init__(self):
        super(JointKBertLoss, self).__init__()

    def backward(self, model_output):
        (model1_output, model2_output) = (model_output[0][0], model_output[1][0])
        model1_output.backward()
        model2_output.backward()

        return (model1_output.item(), model2_output.item())

    def calculate_loss(self, model_output):
        (model1_output, model2_output) = (model_output[0][0], model_output[1][0])
        return (model1_output.item(), model2_output.item())
