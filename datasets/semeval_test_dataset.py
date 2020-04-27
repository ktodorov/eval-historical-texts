import pickle
import os
import numpy as np
import torch
from overrides import overrides

from typing import List

from enums.language import Language

from datasets.dataset_base import DatasetBase
from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.file_service import FileService

from utils import path_utils


class SemEvalTestDataset(DatasetBase):
    def __init__(
            self,
            language: Language,
            arguments_service: ArgumentsServiceBase,
            tokenize_service: BaseTokenizeService,
            file_service: FileService,
            **kwargs):
        super(SemEvalTestDataset, self).__init__()

        self._arguments_service = arguments_service

        challenge_path = file_service.get_challenge_path()
        targets_path = os.path.join(challenge_path, 'eval', str(language), 'targets.txt')

        with open(targets_path, 'r', encoding='utf-8') as targets_file:
            self._target_words = targets_file.read().splitlines()
            self._target_words.sort(key=lambda v: v.upper())

        # English words end with POS tags (e.g. 'test_nn')
        if language == Language.English:
            target_words = [x[:-3] for x in self._target_words]
        else:
            target_words = self._target_words

        encodings = tokenize_service.encode_sequences(target_words)
        self._target_word_ids = [x[0] for x in encodings]

    @overrides
    def __len__(self):
        return len(self._target_word_ids)

    @overrides
    def __getitem__(self, idx):
        return (torch.tensor(self._target_word_ids[idx]).to(self._arguments_service.device), self._target_words[idx])