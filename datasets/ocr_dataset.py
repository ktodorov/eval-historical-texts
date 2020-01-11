import os
import numpy as np
import torch
import pickle

from datasets.dataset_base import DatasetBase
from enums.run_type import RunType
from entities.language_data import LanguageData
from services.arguments_service_base import ArgumentsServiceBase
from services.file_service import FileService
from services.tokenizer_service import TokenizerService
from services.log_service import LogService

from preprocessing.ocr_preprocessing import train_spm_model, preprocess_data, combine_data

from utils import path_utils


class OCRDataset(DatasetBase):
    def __init__(
            self,
            file_service: FileService,
            tokenizer_service: TokenizerService,
            log_service: LogService,
            run_type: RunType,
            language: str,
            device: torch.device,
            vocabulary_size: int,
            reduction: float = None,
            max_articles_length: int = 1000,
            **kwargs):
        super(OCRDataset, self).__init__()

        self._device = device

        output_data_path = file_service.get_data_path()
        language_data_path = os.path.join(
            output_data_path, f'{run_type.to_str()}_language_data.pickle')

        if not tokenizer_service.is_tokenizer_loaded():
            ocr_path = os.path.join('data', 'ocr')
            newseye_path = os.path.join('data', 'newseye')
            trove_path = os.path.join('data', 'trove')
            combine_data(ocr_path, newseye_path, trove_path)

            full_data_path = os.path.join('data', 'ocr', 'full.txt')

            train_spm_model(full_data_path, output_data_path,
                            vocabulary_size)
            tokenizer_service.load_tokenizer_model()

        if not os.path.exists(language_data_path):
            train_data_path = os.path.join('data', 'ocr', 'train')
            test_data_path = os.path.join('data', 'ocr', 'eval')
            preprocess_data(language, train_data_path, test_data_path,
                            output_data_path, tokenizer_service.tokenizer)

        with open(language_data_path, 'rb') as data_file:
            self._language_data: LanguageData = pickle.load(data_file)

            if reduction:
                items_length = int(self._language_data.length * reduction)
                language_data_items = self._language_data.get_entries(
                    items_length)
                self._language_data = LanguageData(
                    language_data_items[0],
                    language_data_items[1],
                    language_data_items[2])

            self._language_data.trim_entries(max_articles_length)
            print(f'Loaded {self._language_data.length} entries for {run_type.to_str()}')
            log_service.log_summary(key=f'\'{run_type.to_str()}\' entries amount', value=self._language_data.length)

    def __len__(self):
        return self._language_data.length

    def __getitem__(self, idx):
        result = self._language_data.get_entry(idx)
        return result

    def use_collate_function(self) -> bool:
        return True

    def collate_function(self, sequences):
        return self._pad_and_sort_batch(sequences)

    def _pad_and_sort_batch(self, DataLoaderBatch):
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        _, sequences, targets = batch_split[0], batch_split[1], batch_split[2]

        lengths = np.array([[len(sequences[i]), len(targets[i])]
                            for i in range(batch_size)])
        max_length = lengths.max(axis=0)

        padded_sequences = np.zeros(
            (batch_size, max_length[0]), dtype=np.int64)
        padded_targets = np.zeros((batch_size, max_length[1]), dtype=np.int64)

        for i, (sequence_length, target_length) in enumerate(lengths):
            padded_sequences[i][0:sequence_length] = sequences[i][0:sequence_length]
            padded_targets[i][0:target_length] = targets[i][0:target_length]

        return self._sort_batch(
            torch.from_numpy(padded_sequences).to(self._device),
            torch.from_numpy(padded_targets).to(self._device),
            torch.tensor(lengths, device=self._device))

    def _sort_batch(self, batch, targets, lengths):
        seq_lengths, perm_idx = lengths[:, 0].sort(0, descending=True)
        seq_tensor = batch[perm_idx]
        targets_tensor = targets[perm_idx]
        return seq_tensor, targets_tensor, seq_lengths
