import pickle
import os
import numpy as np
import torch
from overrides import overrides

from typing import List
from entities.batch_representation import BatchRepresentation

from enums.language import Language

from datasets.dataset_base import DatasetBase
from services.arguments.pretrained_arguments_service import PretrainedArgumentsService
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.file_service import FileService
from services.vocabulary_service import VocabularyService

from utils import path_utils


class SemEvalTestDataset(DatasetBase):
    def __init__(
            self,
            language: Language,
            arguments_service: PretrainedArgumentsService,
            tokenize_service: BaseTokenizeService,
            file_service: FileService,
            vocabulary_service: VocabularyService,
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

        if arguments_service.include_pretrained_model:
            encodings = tokenize_service.encode_sequences(target_words)
            self._target_word_ids = [x[0] for x in encodings]
        else:
            self._target_word_ids = [vocabulary_service.string_to_id(target_word) for target_word in target_words]

    @overrides
    def __len__(self):
        return len(self._target_word_ids)

    @overrides
    def __getitem__(self, idx):
        return self._target_word_ids[idx], self._target_words[idx]

    @overrides
    def use_collate_function(self) -> bool:
        return True

    @overrides
    def collate_function(self, DataLoaderBatch):
        batch_split = list(zip(*DataLoaderBatch))
        word_id, word = batch_split

        batch_representation = BatchRepresentation(
            device=self._arguments_service.device,
            batch_size=1,
            word_sequences=[word_id],
            additional_information=word[0])

        return batch_representation