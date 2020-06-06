import os
import numpy as np
import torch
import pickle
from overrides import overrides

from typing import List
from transformers import BertModel

from datasets.dataset_base import DatasetBase
from enums.run_type import RunType
from entities.language_data import LanguageData
from entities.batch_representation import BatchRepresentation
from services.arguments.postocr_arguments_service import PostOCRArgumentsService
from services.file_service import FileService
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.log_service import LogService
from services.vocabulary_service import VocabularyService
from services.metrics_service import MetricsService
from services.data_service import DataService


class OCRDataset(DatasetBase):
    def __init__(
            self,
            arguments_service: PostOCRArgumentsService,
            file_service: FileService,
            tokenize_service: BaseTokenizeService,
            vocabulary_service: VocabularyService,
            log_service: LogService,
            run_type: RunType):
        super(OCRDataset, self).__init__()

        self._tokenize_service = tokenize_service
        self._vocabulary_service = vocabulary_service

        self._device = arguments_service.device

        # language_data_path = self._get_language_data_path(
        #     file_service,
        #     run_type)

        # self._language_data = self._load_language_data(
        #     log_service,
        #     language_data_path,
        #     run_type,
        #     arguments_service.train_dataset_limit_size if run_type == RunType.Train else arguments_service.validation_dataset_limit_size)

    def _get_language_data_path(
            self,
            file_service: FileService,
            run_type: RunType):
        output_data_path = file_service.get_data_path()
        language_data_path = os.path.join(
            output_data_path, f'{run_type.to_str()}_language_data.pickle')

        if not os.path.exists(language_data_path):
            train_data_path = file_service.get_pickles_path()
            test_data_path = None
            preprocess_data(train_data_path, test_data_path,
                            output_data_path, self._tokenize_service.tokenizer, self._vocabulary_service)

        return language_data_path

    def _load_language_data(
            self,
            log_service: LogService,
            language_data_path: str,
            run_type: RunType,
            reduction: float):

        language_data = LanguageData()
        language_data.load_data(language_data_path)

        total_amount = language_data.length
        if reduction:
            language_data_items = language_data.get_entries(
                reduction)
            language_data = LanguageData(
                language_data_items[0],
                language_data_items[1],
                language_data_items[2],
                language_data_items[3],
                language_data_items[4],
                language_data_items[5],
                language_data_items[6])

        print(
            f'Loaded {language_data.length} entries out of {total_amount} total for {run_type.to_str()}')
        log_service.log_summary(
            key=f'\'{run_type.to_str()}\' entries amount', value=language_data.length)

        return language_data

    @overrides
    def __len__(self):
        return self._language_data.length

    @overrides
    def __getitem__(self, idx):
        result = self._language_data.get_entry(idx)

        _, ocr_aligned, gs_aligned, _, _ = result

        return ocr_aligned, gs_aligned

    @overrides
    def use_collate_function(self) -> bool:
        return True

    @overrides
    def collate_function(self, batch_input):
        batch_size = len(batch_input)
        batch_split = list(zip(*batch_input))

        sequences, targets = batch_split

        batch_representation = BatchRepresentation(
            device=self._device,
            batch_size=batch_size,
            subword_sequences=sequences,
            targets=targets)

        batch_representation.sort_batch()
        return batch_representation