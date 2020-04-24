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
import preprocessing.ocr_download as ocr_download

from utils import path_utils


class OCRCharacterDataset(OCRDataset):
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
            run_type: RunType,
            **kwargs):
        super().__init__(
            arguments_service,
            file_service,
            tokenizer_service,
            vocabulary_service,
            metrics_service,
            log_service,
            pretrained_representations_service,
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
            full_data_path = os.path.join(challenge_path, 'articles')
            if not os.path.exists(full_data_path):
                os.mkdir(full_data_path)

            if len(os.listdir(full_data_path)) == 0:
                newseye_path = os.path.join('data', 'newseye')
                trove_path = os.path.join('data', 'trove')
                ocr_download.combine_data(
                    self._data_service,
                    full_data_path,
                    newseye_path,
                    trove_path)

            pickles_path = file_service.get_pickles_path()
            train_data_path = file_service.get_pickles_path()
            preprocess_data(
                self._tokenizer_service,
                self._metrics_service,
                self._vocabulary_service,
                self._data_service,
                pickles_path,
                full_data_path,
                output_data_path)

        return language_data_path

    @overrides
    def __getitem__(self, idx):
        result = self._language_data.get_entry(idx)

        _, ocr_aligned, _, ocr_text, gs_text, ocr_offsets = result

        return ocr_aligned, ocr_text, gs_text, ocr_offsets

    def _pad_and_sort_batch(self, DataLoaderBatch):
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        sequences, ocr_texts, gs_texts, offset_lists = batch_split

        batch_representation = BaseBatchRepresentation(
            device=self._device,
            batch_size=batch_size,
            subword_sequences=sequences,
            character_sequences=ocr_texts,
            targets=gs_texts,
            offset_lists=offset_lists)

        batch_representation.sort_batch()
        return batch_representation


        # pretrained_representations = self._get_pretrained_representations(
        #     sequences)

        # lengths = np.array([[len(ocr_texts[i]), len(gs_texts[i])]
        #                     for i in range(batch_size)])

        # max_length = lengths.max(axis=0)

        # padded_sequences = np.zeros(
        #     (batch_size, max_length[0]), dtype=np.int64)
        # padded_targets = np.zeros(
        #     (batch_size, max_length[1]), dtype=np.int64) * self._vocabulary_service.pad_token

        # for i, (sequence_length, target_length) in enumerate(lengths):
        #     padded_sequences[i][0:sequence_length] = ocr_texts[i][0:sequence_length]
        #     padded_targets[i][0:target_length] = gs_texts[i][0:target_length]

        # return self._sort_batch(
        #     torch.from_numpy(padded_sequences).to(self._device),
        #     torch.from_numpy(padded_targets).to(self._device),
        #     torch.tensor(lengths, device=self._device),
        #     pretrained_representations,
        #     offset_lists)

    # def _sort_batch(self, batch, targets, lengths, pretrained_embeddings, offset_lists):
    #     seq_lengths, perm_idx = lengths[:, 0].sort(0, descending=True)
    #     seq_tensor = batch[perm_idx]
    #     targets_tensor = targets[perm_idx]

    #     if self._include_pretrained:
    #         pretrained_embeddings = pretrained_embeddings[perm_idx]
    #         offset_lists = [offset_lists[i] for i in perm_idx]

    #     return seq_tensor, targets_tensor, seq_lengths, pretrained_embeddings, offset_lists
