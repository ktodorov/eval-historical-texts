import random
import pickle
import numpy as np
import torch
from typing import List
from overrides import overrides

from datasets.dataset_base import DatasetBase

from services.arguments.arguments_service_base import ArgumentsServiceBase


class JointDataset(DatasetBase):
    def __init__(
            self,
            sub_datasets: List[DatasetBase],
            **kwargs):
        super(JointDataset, self).__init__()

        self._datasets = sub_datasets

    @overrides
    def __len__(self):
        return max([len(dataset) for dataset in self._datasets])

    @overrides
    def __getitem__(self, idx):
        ids = [dataset[self.correct_id(idx, len(dataset))]
               for dataset in self._datasets]

        max_len = 10
        ids = [idx[:max_len] if len(idx) > max_len else idx for idx in ids]
        return ids

    def correct_id(self, idx: int, data_list_length: int) -> int:
        result = idx
        if idx > data_list_length:
            result = random.randint(0, data_list_length)

        return result

    @overrides
    def use_collate_function(self) -> bool:
        return True

    @overrides
    def collate_function(self, sequences):
        if not sequences:
            return []

        result_list = [None for _ in range(len(sequences[0]))]
        entries = [[] for _ in range(len(sequences[0]))]

        for i, sequence in enumerate(sequences):
            for k, entry in enumerate(sequence):
                entries[k].append(entry)

        for k in range(len(sequences[0])):
            result_list[k] = self._datasets[i].collate_function(entries[k])

        return result_list
