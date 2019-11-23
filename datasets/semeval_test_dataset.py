import pickle
import numpy as np
import torch

from typing import List

from datasets.dataset_base import DatasetBase
from services.arguments_service_base import ArgumentsServiceBase
from services.tokenizer_service import TokenizerService

from utils import path_utils


class SemEvalTestDataset(DatasetBase):
    def __init__(
            self,
            language: str,
            arguments_service: ArgumentsServiceBase,
            tokenizer_service: TokenizerService,
            **kwargs):
        super(SemEvalTestDataset, self).__init__()

        targets_path = path_utils.combine_path(
            'data', 'semeval_trial_data', 'targets', f'{language}.txt')

        with open(targets_path, 'r', encoding='utf-8') as targets_file:
            self._target_words = targets_file.read().splitlines()

        self._target_word_ids: List[int] = [tokenizer_service.tokenizer.convert_tokens_to_ids(
            target_word) for target_word in self._target_words]

    def __len__(self):
        return len(self._target_word_ids)

    def __getitem__(self, idx):
        return (self._target_word_ids[idx], self._target_words[idx])
