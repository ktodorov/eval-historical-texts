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
from entities.batch_representations.base_batch_representation import BaseBatchRepresentation
from services.arguments.postocr_arguments_service import PostOCRArgumentsService
from services.file_service import FileService
from services.tokenizer_service import TokenizerService
from services.log_service import LogService
from services.pretrained_representations_service import PretrainedRepresentationsService
from services.vocabulary_service import VocabularyService
from services.metrics_service import MetricsService
from services.data_service import DataService

from preprocessing.ocr_preprocessing import preprocess_data

from utils import path_utils


class OCRDataset(DatasetBase):
    def __init__(
            self,
            arguments_service: PostOCRArgumentsService,
            file_service: FileService,
            tokenizer_service: TokenizerService,
            vocabulary_service: VocabularyService,
            metrics_service: MetricsService,
            log_service: LogService,
            pretrained_representations_service: PretrainedRepresentationsService,
            data_service: DataService,
            run_type: RunType):
        super(OCRDataset, self).__init__()

        self._tokenizer_service = tokenizer_service
        self._metrics_service = metrics_service
        self._vocabulary_service = vocabulary_service
        self._data_service = data_service

        self._device = arguments_service.device
        self._pretrained_representations_service = pretrained_representations_service
        self._include_pretrained = arguments_service.include_pretrained_model
        self._pretrained_model_size = self._pretrained_representations_service.get_pretrained_model_size()
        self._max_length = self._pretrained_representations_service.get_pretrained_max_length()

        language_data_path = self._get_language_data_path(
            file_service,
            run_type)

        self._language_data = self._load_language_data(
            log_service,
            language_data_path,
            run_type,
            arguments_service.train_dataset_limit_size if run_type == RunType.Train else arguments_service.validation_dataset_limit_size)

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
                            output_data_path, self._tokenizer_service.tokenizer, self._vocabulary_service)

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

    def _get_pretrained_representations(
            self,
            ocr_token_lists: List[List[int]]):
        if not self._include_pretrained:
            return []

        batch_size = len(ocr_token_lists)
        lengths = [len(x) for x in ocr_token_lists]
        max_length = max(lengths)
        padded_tokens = np.zeros((batch_size, max_length), dtype=np.int64)

        for i, ocr_token_list in enumerate(ocr_token_lists):
            padded_tokens[i][:lengths[i]] = ocr_token_list

        padded_tokens_tensor = torch.tensor(padded_tokens).to(self._device)
        pretrained_representations = self._pretrained_representations_service.get_pretrained_representation(
            padded_tokens_tensor)
        return pretrained_representations

    @overrides
    def use_collate_function(self) -> bool:
        return True

    @overrides
    def collate_function(self, batch_input):
        batch_size = len(batch_input)
        batch_split = list(zip(*batch_input))

        sequences, targets = batch_split

        batch_representation = BaseBatchRepresentation(
            device=self._device,
            batch_size=batch_size,
            sequences=sequences,
            targets=targets)

        batch_representation.sort_batch()
        return batch_representation