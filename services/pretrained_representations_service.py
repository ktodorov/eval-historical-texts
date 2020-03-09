from transformers import BertModel

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService

class PretrainedRepresentationsService:
    def __init__(
            self,
            arguments_service: PretrainedArgumentsService):

        self._include_pretrained = arguments_service.include_pretrained_model
        self._pretrained_model_size = arguments_service.pretrained_model_size
        self._pretrained_weights = arguments_service.pretrained_weights
        self._pretrained_max_length = arguments_service.pretrained_max_length
        self._pretrained_model = None

        if self._include_pretrained and self._pretrained_model_size and self._pretrained_weights:
            self._pretrained_model = BertModel.from_pretrained(
                arguments_service.pretrained_weights).to(arguments_service.device)
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
