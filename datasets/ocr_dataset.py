import os
import numpy as np
import torch
import pickle

from typing import List
from transformers import BertModel

from datasets.dataset_base import DatasetBase
from enums.run_type import RunType
from entities.language_data import LanguageData
from services.arguments_service_base import ArgumentsServiceBase
from services.file_service import FileService
from services.tokenizer_service import TokenizerService
from services.log_service import LogService
from services.mask_service import MaskService
from services.pretrained_representations_service import PretrainedRepresentationsService

from preprocessing.ocr_preprocessing import train_spm_model, preprocess_data, combine_data

from utils import path_utils


class OCRDataset(DatasetBase):
    def __init__(
            self,
            file_service: FileService,
            tokenizer_service: TokenizerService,
            log_service: LogService,
            mask_service: MaskService,
            pretrained_representations_service: PretrainedRepresentationsService,
            run_type: RunType,
            language: str,
            device: torch.device,
            reduction: float = None,
            max_articles_length: int = 1000,
            include_pretrained: bool = False,
            **kwargs):
        super(OCRDataset, self).__init__()

        self._device = device
        self._mask_service = mask_service
        self._tokenizer_service = tokenizer_service
        self._pretrained_representations_service = pretrained_representations_service
        self._include_pretrained = include_pretrained
        self._pretrained_model_size = self._pretrained_representations_service.get_pretrained_model_size()
        self._max_length = self._pretrained_representations_service.get_pretrained_max_length()

        language_data_path = self._get_language_data_path(
            file_service,
            run_type,
            language)

        self._language_data = self._load_language_data(
            log_service,
            language_data_path,
            run_type,
            reduction,
            max_articles_length)

    def _get_language_data_path(
        self,
        file_service: FileService,
        run_type: RunType,
        language: str):
        output_data_path = file_service.get_data_path()
        language_data_path = os.path.join(
            output_data_path, f'{run_type.to_str()}_language_data.pickle')

        if not os.path.exists(language_data_path):
            train_data_path = os.path.join('data', 'ocr', 'pickles')
            test_data_path = None # os.path.join('data', 'ocr', 'eval')
            preprocess_data(language, train_data_path, test_data_path,
                            output_data_path, self._tokenizer_service.tokenizer)

        return language_data_path

    def _load_language_data(
            self,
            log_service: LogService,
            language_data_path: str,
            run_type: RunType,
            reduction: float,
            max_articles_length: int):
        with open(language_data_path, 'rb') as data_file:
            language_data: LanguageData = pickle.load(data_file)

            if reduction:
                items_length = int(language_data.length * reduction)
                language_data_items = language_data.get_entries(
                    items_length)
                language_data = LanguageData(
                    language_data_items[0],
                    language_data_items[1],
                    language_data_items[2])

            print(
                f'Loaded {language_data.length} entries for {run_type.to_str()}')
            log_service.log_summary(
                key=f'\'{run_type.to_str()}\' entries amount', value=language_data.length)

        return language_data

    def __len__(self):
        return self._language_data.length

    def __getitem__(self, idx):
        result = self._language_data.get_entry(idx)

        _, ocr_aligned, gs_aligned = result

        pretrained_result = self._get_pretrained_representation(ocr_aligned)

        return ocr_aligned, gs_aligned, pretrained_result

    def _get_pretrained_representation(self, ocr_aligned: List[int]):
        if not self._include_pretrained:
            return []

        ocr_aligned_splits = [ocr_aligned]
        if len(ocr_aligned) > self._max_length:
            ocr_aligned_splits = self._split_to_chunks(
                ocr_aligned, chunk_size=self._max_length, overlap_size=2)

        pretrained_outputs = torch.zeros(
            (len(ocr_aligned_splits), self._max_length, self._pretrained_model_size)).to(self._device)

        for i, ocr_aligned_split in enumerate(ocr_aligned_splits):
            ocr_aligned_tensor = torch.Tensor(
                ocr_aligned_split).unsqueeze(0).long().to(self._device)
            pretrained_output = self._pretrained_representations_service.get_pretrained_representation(
                ocr_aligned_tensor)

            _, output_length, _ = pretrained_output.shape

            pretrained_outputs[i, :output_length, :] = pretrained_output

        pretrained_result = pretrained_outputs.view(
            -1, self._pretrained_model_size)

        return pretrained_result

    def _split_to_chunks(self, list_to_split: list, chunk_size: int, overlap_size: int):
        result = [list_to_split[i:i+chunk_size]
                  for i in range(0, len(list_to_split), chunk_size-overlap_size)]
        return result

    def use_collate_function(self) -> bool:
        return True

    def collate_function(self, sequences):
        return self._pad_and_sort_batch(sequences)

    def _pad_and_sort_batch(self, DataLoaderBatch):
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        sequences, targets, pretrained_representations = batch_split

        lengths = np.array([[len(sequences[i]), len(targets[i])]
                            for i in range(batch_size)])

        max_length = lengths.max(axis=0)

        padded_sequences = np.zeros(
            (batch_size, max_length[0]), dtype=np.int64)
        padded_targets = np.zeros((batch_size, max_length[1]), dtype=np.int64)

        padded_pretrained_representations = []
        if self._include_pretrained:
            padded_pretrained_representations = torch.zeros(
                (batch_size, max_length[0], self._pretrained_model_size)).to(self._device)

        for i, (sequence_length, target_length) in enumerate(lengths):
            padded_sequences[i][0:sequence_length] = sequences[i][0:sequence_length]
            padded_targets[i][0:target_length] = targets[i][0:target_length]

            if self._include_pretrained:
                padded_pretrained_representations[i][0:
                                                     sequence_length] = pretrained_representations[i][0:sequence_length]

        return self._sort_batch(
            torch.from_numpy(padded_sequences).to(self._device),
            torch.from_numpy(padded_targets).to(self._device),
            torch.tensor(lengths, device=self._device),
            padded_pretrained_representations)

    def _sort_batch(self, batch, targets, lengths, pretrained_embeddings):
        seq_lengths, perm_idx = lengths[:, 0].sort(0, descending=True)
        seq_tensor = batch[perm_idx]
        targets_tensor = targets[perm_idx]

        if self._include_pretrained:
            pretrained_embeddings = pretrained_embeddings[perm_idx]

        return seq_tensor, targets_tensor, seq_lengths, pretrained_embeddings
