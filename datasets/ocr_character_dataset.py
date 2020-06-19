import os
import numpy as np
import torch
import pickle
from overrides import overrides

from typing import List
from transformers import BertModel

from datasets.ocr_dataset import OCRDataset
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
from services.process.ocr_character_process_service import OCRCharacterProcessService


class OCRCharacterDataset(OCRDataset):
    def __init__(
            self,
            arguments_service: PostOCRArgumentsService,
            file_service: FileService,
            tokenize_service: BaseTokenizeService,
            vocabulary_service: VocabularyService,
            metrics_service: MetricsService,
            log_service: LogService,
            data_service: DataService,
            process_service: OCRCharacterProcessService,
            run_type: RunType,
            **kwargs):
        super().__init__(
            arguments_service,
            file_service,
            tokenize_service,
            vocabulary_service,
            log_service,
            run_type,
            **kwargs)

        self._process_service = process_service
        self._tokenize_service = tokenize_service
        self._run_type = run_type

        self._language_data = process_service.get_language_data(run_type)

    @overrides
    def __getitem__(self, idx):
        result = self._language_data.get_entry(idx)

        _, ocr_aligned, _, ocr_text, gs_text, ocr_offsets = result
        filtered_tokens = [token.replace('#', '') for token in self._tokenize_service.decode_tokens(ocr_aligned[1:])]

        return ocr_aligned[1:], ocr_text[1:], gs_text, ocr_offsets, filtered_tokens

    @overrides
    def collate_function(self, batch_input):
        batch_size = len(batch_input)
        batch_split = list(zip(*batch_input))

        sequences, ocr_texts, gs_texts, offset_lists, tokens = batch_split

        batch_representation = BatchRepresentation(
            device=self._device,
            batch_size=batch_size,
            subword_sequences=sequences,
            character_sequences=ocr_texts,
            targets=gs_texts,
            tokens=tokens,
            offset_lists=offset_lists)

        batch_representation.sort_batch()
        return batch_representation