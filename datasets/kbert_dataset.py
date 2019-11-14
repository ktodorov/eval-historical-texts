import numpy as np
import torch
import pickle

from datasets.dataset_base import DatasetBase
from services.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService
from services.mask_service import MaskService

from utils import path_utils


class KBertDataset(DatasetBase):
    def __init__(
            self,
            language: str,
            arguments_service: ArgumentsServiceBase,
            mask_service: MaskService,
            corpus_id: int = 1,
            **kwargs):
        super(KBertDataset, self).__init__()

        self._mask_service = mask_service
        self._arguments_service = arguments_service

        data_folder = path_utils.combine_path(
            'data', 'semeval_trial_data', 'ids', language, f'ids{corpus_id}.pickle')
        with open(data_folder, 'rb') as data_file:
            self._ids = pickle.load(data_file)

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx):
        result = self._ids[idx]
        return result

    def use_collate_function(self) -> bool:
        return True

    def collate_function(self, sequences):
        return self._pad_and_sort_batch(sequences)

    def _pad_and_sort_batch(self, sequences):
        batch_size = len(sequences)

        lengths = [len(sequence) for sequence in sequences]
        max_length = max(lengths)

        padded_sequences = np.ones((batch_size, max_length), dtype=np.int64)

        for i, l in enumerate(lengths):
            padded_sequences[i][0:l] = sequences[i][0:l]

        return self._sort_batch(
            torch.from_numpy(padded_sequences).to(
                self._arguments_service.get_argument('device')),
            torch.tensor(lengths).to(self._arguments_service.get_argument('device')))

    def _sort_batch(self, batch, lengths):
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = batch[perm_idx]
        return self._mask_service.mask_tokens(seq_tensor)
