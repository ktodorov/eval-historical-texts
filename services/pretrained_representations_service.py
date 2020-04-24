import torch
import os
from transformers import BertModel
import fasttext

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService
from services.file_service import FileService


class PretrainedRepresentationsService:
    def __init__(
            self,
            arguments_service: PretrainedArgumentsService,
            file_service: FileService):

        self._arguments_service = arguments_service
        self._device = arguments_service.device

        self._include_pretrained = arguments_service.include_pretrained_model
        self._pretrained_model_size = arguments_service.pretrained_model_size
        self._pretrained_weights = arguments_service.pretrained_weights
        self._pretrained_max_length = arguments_service.pretrained_max_length
        self._pretrained_model = None

        self._include_fasttext_model = arguments_service.include_fasttext_model

        if self._include_pretrained and self._pretrained_model_size and self._pretrained_weights:
            self._pretrained_model = BertModel.from_pretrained(
                arguments_service.pretrained_weights).to(arguments_service.device)
            self._pretrained_model.eval()

            for param in self._pretrained_model.parameters():
                param.requires_grad = False

        if self._include_fasttext_model:
            assert arguments_service.fasttext_model is not None, 'fast text model is not supplied when include-fasttext-model is set to true'

            data_path = file_service.get_data_path()
            fasttext_path = os.path.join(data_path, 'fasttext', self._arguments_service.fasttext_model)
            assert os.path.exists(fasttext_path), f'fast text model not found in {fasttext_path}'


            self._fasttext_dimension = self._arguments_service.fasttext_model_size
            self._fasttext_model = fasttext.load_model(fasttext_path)

    def get_pretrained_representation(self, input):
        if self.pretrained_model is None:
            return []

        output = self._pretrained_model.forward(input)
        return output[0]

    def get_fasttext_representation(
            self,
            sequences_strings):
        batch_size = len(sequences_strings)
        sequence_length = max([len(s) for s in sequences_strings])

        fasttext_tensor = torch.zeros(
            (batch_size, sequence_length, self._fasttext_dimension)).to(self._device)
        for b in range(batch_size):
            current_string = sequences_strings[b]
            for i, token in enumerate(current_string):
                if token.startswith('##'):
                    token = token[2:]

                fasttext_representation = self._fasttext_model.get_word_vector(token)
                fasttext_tensor[b, i, :] = torch.Tensor(fasttext_representation).to(self._device)

        return fasttext_tensor

    def get_pretrained_model_size(self) -> int:
        return self._pretrained_model_size

    def get_pretrained_max_length(self) -> int:
        return self._pretrained_max_length