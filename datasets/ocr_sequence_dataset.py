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

from preprocessing.ocr_preprocessing import preprocess_data
import preprocessing.ocr_download as ocr_download

from utils import path_utils


class OCRSequenceDataset(OCRDataset):
    def __init__(
            self,
            arguments_service: PostOCRArgumentsService,
            file_service: FileService,
            tokenize_service: BaseTokenizeService,
            vocabulary_service: VocabularyService,
            metrics_service: MetricsService,
            log_service: LogService,
            data_service: DataService,
            run_type: RunType,
            **kwargs):
        super().__init__(
            arguments_service,
            file_service,
            tokenize_service,
            vocabulary_service,
            metrics_service,
            log_service,
            data_service,
            run_type,
            **kwargs)

    @overrides
    def _get_language_data_path(
        self,
        file_service: FileService,
        run_type: RunType):
        output_data_path = file_service.get_data_path()
        language_data_path = os.path.join(
            output_data_path, f'{run_type.to_str()}_language_data.pickle')

        if not os.path.exists(language_data_path):
            challenge_path = file_service.get_challenge_path()
            full_data_path = os.path.join(challenge_path, 'full')
            if not os.path.exists(full_data_path) or len(os.listdir(full_data_path)) == 0:
                newseye_path = os.path.join('data', 'newseye')
                trove_path = os.path.join('data', 'trove')
                ocr_download.combine_data(challenge_path, newseye_path, trove_path)

            pickles_path = file_service.get_pickles_path()
            train_data_path = file_service.get_pickles_path()
            preprocess_data(
                self._tokenize_service,
                self._metrics_service,
                self._vocabulary_service,
                pickles_path,
                full_data_path,
                output_data_path)

        return language_data_path

    @overrides
    def __getitem__(self, idx):
        result = self._language_data.get_entry(idx)

        _, ocr_aligned, gs_aligned, ocr_text, gs_text = result

        return ocr_aligned[1:], gs_aligned, ocr_text[1:], gs_text

    def _pad_and_sort_batch(self, DataLoaderBatch):
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        sequences, _, ocr_texts, gs_texts = batch_split

        batch_representation = BatchRepresentation(
            device=self._device,
            batch_size=batch_size,
            subword_sequences=sequences,
            character_sequences=ocr_texts,
            targets=gs_texts,
            offset_lists=None) # TODO add offset lists

        batch_representation.sort_batch()
        return batch_representation