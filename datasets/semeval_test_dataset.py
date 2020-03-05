import pickle
import os
import numpy as np
import torch

from typing import List

from enums.language import Language

from datasets.dataset_base import DatasetBase
from services.arguments_service_base import ArgumentsServiceBase
from services.tokenizer_service import TokenizerService
from services.file_service import FileService

from utils import path_utils


class SemEvalTestDataset(DatasetBase):
    def __init__(
            self,
            language: Language,
            arguments_service: ArgumentsServiceBase,
            tokenizer_service: TokenizerService,
            file_service: FileService,
            **kwargs):
        super(SemEvalTestDataset, self).__init__()

        self._arguments_service = arguments_service

        challenge_path = file_service.get_challenge_path()
        targets_path = os.path.join(challenge_path, 'eval', str(language), 'targets.txt')

        with open(targets_path, 'r', encoding='utf-8') as targets_file:
            self._target_words = targets_file.read().splitlines()

        # English words end with POS tags (e.g. 'test_nn')
        if language == Language.English:
            self._target_words = [x[:-3] for x in self._target_words]

        encodings = tokenizer_service.encode_sequences(self._target_words)
        self._target_word_ids = [x[0] for x in encodings]

    def __len__(self):
        return len(self._target_word_ids)

    def __getitem__(self, idx):
        return (torch.tensor(self._target_word_ids[idx]).to(self._arguments_service.device), self._target_words[idx])