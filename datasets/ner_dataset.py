import os
import numpy as np

import torch

from typing import List

from entities.ne_line import NELine
from entities.ne_collection import NECollection

from datasets.dataset_base import DatasetBase
from enums.run_type import RunType
from enums.language import Language
from preprocessing.ner_preprocessing import preprocess_data

from services.file_service import FileService
from services.tokenizer_service import TokenizerService
from services.pretrained_representations_service import PretrainedRepresentationsService


class NERDataset(DatasetBase):
    def __init__(
            self,
            file_service: FileService,
            tokenizer_service: TokenizerService,
            pretrained_representations_service: PretrainedRepresentationsService,
            run_type: RunType,
            language: Language,
            device: torch.device,
            max_articles_length: int = 1000,
            include_pretrained: bool = False,
            limit_size: int = None):
        super().__init__()

        self._device = device
        self._pretrained_representations_service = pretrained_representations_service
        self._include_pretrained = include_pretrained
        self._pretrained_model_size = self._pretrained_representations_service.get_pretrained_model_size()
        self._max_length = self._pretrained_representations_service.get_pretrained_max_length()

        data_path = file_service.get_data_path()
        file_suffix = 'train' if run_type == RunType.Train else 'dev'
        language_suffix = self._get_language_suffix(language)
        file_path = os.path.join(data_path, f'HIPE-data-v0.9-{file_suffix}-{language_suffix}.tsv')
        self.ne_collection = preprocess_data(file_path, tokenizer_service, limit_size)

        print(f'Loaded {len(self.ne_collection)} items for \'{run_type}\' set')

    def __len__(self):
        return len(self.ne_collection)

    def __getitem__(self, idx):
        item: NELine = self.ne_collection[idx]

        entities = {
            None: 0,
            'B-org': 1,
            'I-org': 2,
            'B-pers': 3,
            'I-pers': 4,
            'B-loc': 5,
            'B-time': 6,
            'I-time': 7,
            'I-loc': 8,
            'B-prod': 9,
            'I-prod': 10,
        }

        coarse_entity_labels = [entities[x] for x in item.ne_coarse_lit]

        pretrained_result = self._get_pretrained_representation(item.token_ids)

        return item.token_ids, coarse_entity_labels, pretrained_result

    def _get_pretrained_representation(self, token_ids: List[int]):
        if not self._include_pretrained:
            return []

        token_ids_splits = [token_ids]
        if len(token_ids) > self._max_length:
            token_ids_splits = self._split_to_chunks(
                token_ids, chunk_size=self._max_length, overlap_size=2)

        pretrained_outputs = torch.zeros(
            (len(token_ids_splits), self._max_length, self._pretrained_model_size)).to(self._device) * -1

        for i, token_ids_split in enumerate(token_ids_splits):
            token_ids_tensor = torch.Tensor(
                token_ids_split).unsqueeze(0).long().to(self._device)
            pretrained_output = self._pretrained_representations_service.get_pretrained_representation(
                token_ids_tensor)

            _, output_length, _ = pretrained_output.shape

            pretrained_outputs[i, :output_length, :] = pretrained_output

        pretrained_result = pretrained_outputs.view(
            -1, self._pretrained_model_size)

        return pretrained_result


    def _get_language_suffix(self, language: Language):
        if language == Language.English:
            return 'en'
        elif language == Language.French:
            return 'fr'
        elif language == Language.German:
            return 'de'
        else:
            raise Exception('Unsupported language')

    def use_collate_function(self) -> bool:
        return True

    def collate_function(self, sequences):
        return self._pad_and_sort_batch(sequences)

    def _pad_and_sort_batch(self, DataLoaderBatch):
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        sequences, targets, pretrained_representations = batch_split

        lengths = [len(sequence) for sequence in sequences]

        max_length = max(lengths)

        padded_sequences = np.zeros(
            (batch_size, max_length), dtype=np.int64) * -1
        padded_targets = np.zeros((batch_size, max_length), dtype=np.int64) * -1

        padded_pretrained_representations = []
        if self._include_pretrained:
            padded_pretrained_representations = torch.zeros(
                (batch_size, max_length, self._pretrained_model_size)).to(self._device) * -1

        for i, sequence_length in enumerate(lengths):
            padded_sequences[i][0:sequence_length] = sequences[i][0:sequence_length]
            padded_targets[i][0:sequence_length] = targets[i][0:sequence_length]

            if self._include_pretrained:
                padded_pretrained_representations[i][0:
                                                     sequence_length] = pretrained_representations[i][0:sequence_length]

        return self._sort_batch(
            torch.from_numpy(padded_sequences).to(self._device),
            torch.from_numpy(padded_targets).to(self._device),
            torch.tensor(lengths, device=self._device),
            padded_pretrained_representations)

    def _sort_batch(self, batch, targets, lengths, pretrained_embeddings):
        seq_lengths, perm_idx = lengths.sort(descending=True)
        seq_tensor = batch[perm_idx]
        targets_tensor = targets[perm_idx]

        if self._include_pretrained:
            pretrained_embeddings = pretrained_embeddings[perm_idx]

        return seq_tensor, targets_tensor, seq_lengths, pretrained_embeddings

