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

from preprocessing.semeval_preprocessing import preprocess_data

from utils import path_utils


class SemanticChangeDataset(DatasetBase):
    def __init__(
            self,
            language: str,
            arguments_service: SemanticArgumentsService,
            mask_service: MaskService,
            file_service: FileService,
            tokenizer_service: TokenizerService,
            log_service: LogService,
            corpus_id: int = None,
            **kwargs):
        super().__init__()

        self._mask_service = mask_service
        self._arguments_service = arguments_service

        if not corpus_id:
            corpus_id = arguments_service.corpus

        full_data_path = file_service.get_data_path()
        ids_path = os.path.join(full_data_path, f'ids{corpus_id}.pickle')

        if not os.path.exists(ids_path):
            challenge_path = file_service.get_challenge_path()
            semeval_data_path = os.path.join(challenge_path, 'eval')
            pretrained_weights = self._arguments_service.pretrained_weights

            preprocess_data(
                corpus_id,
                language,
                semeval_data_path,
                full_data_path,
                tokenizer_service)

        with open(ids_path, 'rb') as data_file:
            self._ids = pickle.load(data_file)

            if arguments_service.pretrained_max_length:
                self._ids = [x for x in self._ids if len(x[0]) < arguments_service.pretrained_max_length]

            reduction = arguments_service.train_dataset_limit_size
            if reduction:
                items_length = int(len(self._ids) * reduction)
                self._ids = self._ids[:items_length]

            print(f'Loaded {len(self._ids)} entries')
            log_service.log_summary(key='Entries amount', value=len(self._ids))

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx):
        token_ids, mask = self._ids[idx]
        return token_ids, mask

    def use_collate_function(self) -> bool:
        return True

    def collate_function(self, sequences):
        return self._pad_and_sort_batch(sequences)

    def _pad_and_sort_batch(self, DataLoaderBatch):
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        tokens_ids, masks = batch_split
        lengths = [len(sequence) for sequence in tokens_ids]
        max_length = max(lengths)

        padded_sequences = np.zeros((batch_size, max_length), dtype=np.int64)
        padded_masks = np.ones((batch_size, max_length), dtype=np.int64)

        for i, l in enumerate(lengths):
            padded_sequences[i][0:l] = tokens_ids[i][0:l]
            padded_masks[i][0:l] = masks[i][0:l]

        return self._sort_batch(
            torch.from_numpy(padded_sequences).to(
                self._arguments_service.device),
            torch.from_numpy(padded_masks).bool().to(
                self._arguments_service.device),
            torch.tensor(lengths).to(self._arguments_service.device))

    def _sort_batch(self, sequences, masks, lengths):
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = sequences[perm_idx]
        mask_tensor = masks[perm_idx]
        return self._mask_service.mask_tokens(seq_tensor, masks, seq_lengths)
