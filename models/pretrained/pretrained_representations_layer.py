import os
from transformers import PreTrainedModel, BertModel, CamembertModel
import fasttext

import torch
from torch import nn

from overrides import overrides

from services.file_service import FileService

from entities.options.pretrained_representations_options import PretrainedRepresentationsOptions

from enums.pretrained_model import PretrainedModel

from models.model_base import ModelBase

class PretrainedRepresentationsLayer(ModelBase):
    def __init__(
            self,
            file_service: FileService,
            device: str,
            pretrained_representations_options: PretrainedRepresentationsOptions):
        super().__init__()

        self._device = device
        self.do_not_save: bool = (not pretrained_representations_options.fine_tune_pretrained and
                                  not pretrained_representations_options.fine_tune_after_convergence)

        self._include_pretrained = pretrained_representations_options.include_pretrained_model
        self._pretrained_model_size = pretrained_representations_options.pretrained_model_size
        self._pretrained_weights = pretrained_representations_options.pretrained_weights
        self._pretrained_max_length = pretrained_representations_options.pretrained_max_length
        self._pretrained_model: PreTrainedModel = None

        self._fine_tune_pretrained = pretrained_representations_options.fine_tune_pretrained
        self._fine_tune_after_convergence = pretrained_representations_options.fine_tune_after_convergence

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

        if self._include_fasttext_model:
            assert pretrained_representations_options.fasttext_model is not None, 'fast text model is not supplied when include-fasttext-model is set to true'

            data_path = file_service.get_initial_data_path()
            fasttext_path = os.path.join(
                data_path, 'fasttext', pretrained_representations_options.fasttext_model)
            assert os.path.exists(
                fasttext_path), f'fast text model not found in {fasttext_path}'

            self._fasttext_dimension = pretrained_representations_options.fasttext_model_size
            self._fasttext_model = fasttext.load_model(fasttext_path)

    def get_pretrained_representation(self, input):
        if self._pretrained_model is None:
            return []

        if input.shape[1] > self._pretrained_max_length:
            overlap_size = 5
            window_size = self._pretrained_max_length - (overlap_size * 2)
            offset_pairs = self.get_split_indices(input.shape[1], window_size, overlap_size)
            result_tensor = torch.zeros(input.shape[0], input.shape[1], self._pretrained_model_size, device=input.device)

            for (start_offset, end_offset) in offset_pairs:
                current_input = input[:, start_offset:end_offset]
                current_output = self._pretrained_model.forward(current_input)
                current_representations = current_output[0]

                if start_offset > 0:
                    result_tensor[:, start_offset+overlap_size:end_offset] = current_representations[:, overlap_size:]
                    # we get the mean of the overlapping representations
                    result_tensor[:, start_offset:start_offset+overlap_size] = torch.mean(
                        torch.stack([
                            result_tensor[:, start_offset:start_offset+overlap_size],
                            current_representations[:, :overlap_size]]))
                else:
                    result_tensor[:, :end_offset] = current_representations

        else:
            output = self._pretrained_model.forward(input)
            result_tensor = output[0]

        return result_tensor

    def get_split_indices(self, full_length: int, window_size: int, overlap_size=5):

        offset_pairs = []
        for position in range(0, full_length, window_size-overlap_size):
            start_offset = position
            end_offset = position + window_size

            if end_offset > full_length:
                end_offset = full_length

            offset_pairs.append((start_offset, end_offset))

            if end_offset >= full_length:
                break

        return offset_pairs

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

                fasttext_representation = self._fasttext_model.get_word_vector(
                    token)
                fasttext_tensor[b, i, :] = torch.Tensor(
                    fasttext_representation).to(self._device)

        return fasttext_tensor

    @property
    @overrides
    def keep_frozen(self) -> bool:
        return not self._fine_tune_pretrained


    @overrides
    def on_convergence(self) -> bool:
        if self._fine_tune_after_convergence and not self._fine_tune_pretrained:
            print('Starting to fine-tune pre-trained...')
            self._fine_tune_pretrained = True
            return True

        return False