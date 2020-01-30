from transformers import BertModel

class PretrainedRepresentationsService:
    def __init__(
            self,
            include_pretrained: bool,
            pretrained_model_size: int,
            pretrained_weights: str,
            pretrained_max_length: int,
            device: str):

        self._include_pretrained = include_pretrained
        self._pretrained_model_size = pretrained_model_size
        self._pretrained_weights = pretrained_weights
        self._pretrained_max_length = pretrained_max_length
        self._pretrained_model = None

        if self._include_pretrained and self._pretrained_model_size and self._pretrained_weights:
            self._pretrained_model = BertModel.from_pretrained(
                pretrained_weights).to(device)
            self._pretrained_model.eval()

            for param in self._pretrained_model.parameters():
                param.requires_grad = False

    def get_pretrained_representation(self, input):
        output = self._pretrained_model.forward(input)
        return output[0]

    def get_pretrained_model_size(self) -> int:
        return self._pretrained_model_size

    def get_pretrained_max_length(self) -> int:
        return self._pretrained_max_length
