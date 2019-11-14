import random
import pickle
import numpy as np
import torch

from datasets.dataset_base import DatasetBase
from services.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService
from services.mask_service import MaskService

from utils import path_utils


class JointKBertDataset(DatasetBase):
    def __init__(
            self,
            language: str,
            arguments_service: ArgumentsServiceBase,
            mask_service: MaskService,
            **kwargs):
        super(JointKBertDataset, self).__init__()

        self._mask_service = mask_service
        self._arguments_service = arguments_service

        data_folder = path_utils.combine_path(
            'data', 'semeval_trial_data', 'ids', language)
        corpus_path_1 = path_utils.combine_path(data_folder, 'ids1.pickle')
        corpus_path_2 = path_utils.combine_path(data_folder, 'ids2.pickle')

        with open(corpus_path_1, 'rb') as data_file:
            self._ids1 = pickle.load(data_file)

        with open(corpus_path_2, 'rb') as data_file:
            self._ids2 = pickle.load(data_file)

    def __len__(self):
        return max(len(self._ids1), len(self._ids2))

    def __getitem__(self, idx):
        id1 = self.correct_id(idx, self._ids1)
        id2 = self.correct_id(idx, self._ids2)

        result_ids = [self._ids1[id1], self._ids2[id2]]

        max_len = 10
        if len(result_ids[0]) > max_len:
            result_ids[0] = result_ids[0][:max_len]

        if len(result_ids[1]) > max_len:
            result_ids[1] = result_ids[1][:max_len]

        result = (result_ids[0], result_ids[1])
        return result

    def correct_id(self, idx: int, data_list: list) -> int:
        result = idx
        if idx > len(data_list):
            result = random.randint(0, len(data_list))

        return result

    def use_collate_function(self) -> bool:
        return True

    def collate_function(self, sequences):
        return self._pad_and_sort_batch(sequences)

    def _pad_and_sort_batch(self, sequences):
        batch_size = len(sequences)

        lengths = np.array([[len(sequence[0]), len(sequence[1])]
                            for sequence in sequences])

        max_length1 = max(lengths[:, 0])
        max_length2 = max(lengths[:, 1])

        padded_sequences1 = np.ones((batch_size, max_length1), dtype=np.int64)
        padded_sequences2 = np.ones((batch_size, max_length2), dtype=np.int64)

        for i, l in enumerate(lengths[:, 0]):
            padded_sequences1[i][0:l] = sequences[i][0:l][0]

        for i, l in enumerate(lengths[:, 1]):
            padded_sequences2[i][0:l] = sequences[i][0:l][1]

        return (self._sort_batch(
            torch.from_numpy(padded_sequences1).to(
                self._arguments_service.get_argument('device')),
            torch.tensor(lengths[:, 0]).to(self._arguments_service.get_argument('device'))),
            self._sort_batch(
            torch.from_numpy(padded_sequences2).to(
                self._arguments_service.get_argument('device')),
            torch.tensor(lengths[:, 1]).to(self._arguments_service.get_argument('device'))))

    def _sort_batch(self, batch, lengths):
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = batch[perm_idx]
        return self._mask_service.mask_tokens(seq_tensor)
