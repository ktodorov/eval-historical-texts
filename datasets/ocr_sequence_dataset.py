import os
import numpy as np
import torch
import pickle

from typing import List
from transformers import BertModel

from datasets.ocr_dataset import OCRDataset
from enums.run_type import RunType
from entities.language_data import LanguageData
from services.arguments_service_base import ArgumentsServiceBase
from services.file_service import FileService
from services.tokenizer_service import TokenizerService
from services.log_service import LogService
from services.pretrained_representations_service import PretrainedRepresentationsService
from services.vocabulary_service import VocabularyService

from preprocessing.ocr_preprocessing import train_spm_model, preprocess_data, combine_data

from utils import path_utils


class OCRSequenceDataset(OCRDataset):
    def __init__(
            self,
            file_service: FileService,
            tokenizer_service: TokenizerService,
            vocabulary_service: VocabularyService,
            log_service: LogService,
            pretrained_representations_service: PretrainedRepresentationsService,
            run_type: RunType,
            language: str,
            device: torch.device,
            reduction: float = None,
            max_articles_length: int = 1000,
            include_pretrained: bool = False,
            **kwargs):
        super(OCRSequenceDataset, self).__init__(
            file_service,
            tokenizer_service,
            log_service,
            pretrained_representations_service,
            run_type,
            language,
            device,
            reduction,
            max_articles_length,
            include_pretrained,
            **kwargs)

        self._vocabulary_service = vocabulary_service

    def _get_language_data_path(
        self,
        file_service: FileService,
        run_type: RunType,
        language: str):
        output_data_path = file_service.get_data_path()
        language_data_path = os.path.join(
            output_data_path, f'{run_type.to_str()}_language_data.pickle')

        if not os.path.exists(language_data_path):
            train_data_path = file_service.get_pickles_path()
            test_data_path = None # os.path.join('data', 'ocr', 'eval')
            preprocess_data(language, train_data_path, test_data_path,
                            output_data_path, self._tokenizer_service.tokenizer, self._vocabulary_service)

        return language_data_path

    def __getitem__(self, idx):
        result = self._language_data.get_entry(idx)

        _, ocr_aligned, gs_aligned, ocr_text, gs_text = result

        pretrained_result = self._get_pretrained_representation(ocr_aligned)

        return ocr_aligned, gs_aligned, ocr_text, gs_text, pretrained_result

    def _pad_and_sort_batch(self, DataLoaderBatch):
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        sequences, targets, ocr_texts, gs_texts, pretrained_representations = batch_split

        lengths = np.array([[len(sequences[i]), len(gs_texts[i])]
                            for i in range(batch_size)])

        max_length = lengths.max(axis=0)

        padded_sequences = np.zeros(
            (batch_size, max_length[0]), dtype=np.int64)
        padded_targets = np.zeros((batch_size, max_length[1]), dtype=np.int64) * self._vocabulary_service.pad_token
        padded_ocr_texts = np.zeros((batch_size, max([len(x) for x in ocr_texts])), dtype=np.int64) * self._vocabulary_service.pad_token

        padded_pretrained_representations = []
        if self._include_pretrained:
            padded_pretrained_representations = torch.zeros(
                (batch_size, max_length[0], self._pretrained_model_size)).to(self._device)


        for i, (sequence_length, target_length) in enumerate(lengths):
            padded_sequences[i][0:sequence_length] = sequences[i][0:sequence_length]
            padded_targets[i][0:target_length] = gs_texts[i][0:target_length]
            padded_ocr_texts[i][0:len(ocr_texts[i])] = ocr_texts[i]

            if self._include_pretrained:
                padded_pretrained_representations[i][0:
                                                     sequence_length] = pretrained_representations[i][0:sequence_length]

        return self._sort_batch(
            torch.from_numpy(padded_sequences).to(self._device),
            torch.from_numpy(padded_targets).to(self._device),
            torch.tensor(lengths, device=self._device),
            padded_pretrained_representations,
            torch.from_numpy(padded_ocr_texts).to(self._device))

    def _sort_batch(self, batch, targets, lengths, pretrained_embeddings, ocr_texts):
        seq_lengths, perm_idx = lengths[:, 0].sort(0, descending=True)
        seq_tensor = batch[perm_idx]
        targets_tensor = targets[perm_idx]
        ocr_texts = ocr_texts[perm_idx]

        if self._include_pretrained:
            pretrained_embeddings = pretrained_embeddings[perm_idx]

        return seq_tensor, targets_tensor, seq_lengths, pretrained_embeddings, ocr_texts
