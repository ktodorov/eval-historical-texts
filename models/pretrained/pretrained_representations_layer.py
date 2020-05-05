import os
from transformers import BertModel, CamembertModel
import fasttext

import torch
from torch import nn

from overrides import overrides

from services.file_service import FileService

from entities.options.pretrained_representations_options import PretrainedRepresentationsOptions

from enums.pretrained_model import PretrainedModel

class PretrainedRepresentationsLayer(nn.Module):
    def __init__(
            self,
            file_service: FileService,
            device: str,
            pretrained_representations_options: PretrainedRepresentationsOptions):
        super().__init__()

        self._device = device

        self._include_pretrained = pretrained_representations_options.include_pretrained_model
        self._pretrained_model_size = pretrained_representations_options.pretrained_model_size
        self._pretrained_weights = pretrained_representations_options.pretrained_weights
        self._pretrained_max_length = pretrained_representations_options.pretrained_max_length
        self._pretrained_model = None

        self._fine_tune_pretrained = pretrained_representations_options.fine_tune_pretrained

        self._include_fasttext_model = pretrained_representations_options.include_fasttext_model

        if self._include_pretrained and self._pretrained_model_size and self._pretrained_weights:
            if pretrained_representations_options.pretrained_model == PretrainedModel.BERT:
                self._pretrained_model = BertModel.from_pretrained(
                    pretrained_representations_options.pretrained_weights)
            elif pretrained_representations_options.pretrained_model == PretrainedModel.CamemBERT:
                self._pretrained_model = CamembertModel.from_pretrained(
                    pretrained_representations_options.pretrained_weights)

            if pretrained_representations_options.fine_tune_pretrained:
                self._pretrained_model.train()
            else:
                self._pretrained_model.eval()

            for param in self._pretrained_model.parameters():
                param.requires_grad = False

        if self._include_fasttext_model:
            assert pretrained_representations_options.fasttext_model is not None, 'fast text model is not supplied when include-fasttext-model is set to true'

            data_path = file_service.get_data_path()
            fasttext_path = os.path.join(data_path, 'fasttext', pretrained_representations_options.fasttext_model)
            assert os.path.exists(fasttext_path), f'fast text model not found in {fasttext_path}'


            self._fasttext_dimension = pretrained_representations_options.fasttext_model_size
            self._fasttext_model = fasttext.load_model(fasttext_path)

    def get_pretrained_representation(self, input):
        if self._pretrained_model is None:
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

    @overrides
    def train(self, mode=True):
        # If fine-tuning is disabled, we don't set the module to train mode
        if mode and not self._fine_tune_pretrained:
            return

        super().train(mode)

    @overrides
    def eval(self):
        super().eval()