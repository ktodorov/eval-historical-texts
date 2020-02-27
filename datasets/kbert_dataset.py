import os
import numpy as np
import torch
import pickle

from datasets.dataset_base import DatasetBase
from services.semantic_arguments_service import SemanticArgumentsService
from services.file_service import FileService
from services.mask_service import MaskService
from services.tokenizer_service import TokenizerService
from services.log_service import LogService

from preprocessing.semeval_dataset import preprocess_data

from utils import path_utils


class KBertDataset(DatasetBase):
    def __init__(
            self,
            language: str,
            arguments_service: SemanticArgumentsService,
            mask_service: MaskService,
            file_service: FileService,
            tokenizer_service: TokenizerService,
            log_service: LogService,
            corpus_id: int = 1,
            **kwargs):
        super(KBertDataset, self).__init__()

        self._mask_service = mask_service
        self._arguments_service = arguments_service

        full_data_path = file_service.get_data_path()
        ids_path = os.path.join(full_data_path, f'ids{corpus_id}.pickle')

        if not os.path.exists(ids_path):
            semeval_data_path = os.path.join('data', 'semeval_trial_data')
            pretrained_weights = self._arguments_service.pretrained_weights

            tokenizer = tokenizer_service.tokenizer
            preprocess_data(corpus_id, language, semeval_data_path,
                            full_data_path, pretrained_weights, tokenizer)

        with open(ids_path, 'rb') as data_file:
            self._ids = pickle.load(data_file)

            reduction = arguments_service.train_dataset_limit_size
            if reduction:
                items_length = int(len(self._ids) * reduction)
                self._ids = self._ids[:items_length]

            print(f'Loaded {len(self._ids)} entries')
            log_service.log_summary(key='Entries amount', value=len(self._ids))

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx):
        result = self._ids[idx]
        return result

    def use_collate_function(self) -> bool:
        return True

    def collate_function(self, sequences):
        return self._pad_and_sort_batch(sequences)

    def _pad_and_sort_batch(self, sequences):
        batch_size = len(sequences)

        lengths = [len(sequence) for sequence in sequences]
        max_length = max(lengths)

        padded_sequences = np.ones((batch_size, max_length), dtype=np.int64)

        for i, l in enumerate(lengths):
            padded_sequences[i][0:l] = sequences[i][0:l]

        return self._sort_batch(
            torch.from_numpy(padded_sequences).to(
                self._arguments_service.device),
            torch.tensor(lengths).to(self._arguments_service.device))

    def _sort_batch(self, batch, lengths):
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = batch[perm_idx]
        return self._mask_service.mask_tokens(seq_tensor, seq_lengths)
