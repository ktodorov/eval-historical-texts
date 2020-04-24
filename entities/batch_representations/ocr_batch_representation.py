import numpy as np
import torch

from overrides import overrides

from entities.batch_representations.base_batch_representation import BaseBatchRepresentation


class OCRBatchRepresentation(BaseBatchRepresentation):
    def __init__(
            self,
            device: str,
            batch_size: int,
            tokenized_sequences: list,
            ocr_texts: list,
            gs_texts: list,
            offset_lists: list,
            pad_idx: int = 0):
        super().__init__(
            device,
            batch_size,
            ocr_texts,
            gs_texts,
            offset_lists=offset_lists,
            pad_idx=pad_idx)

        self._tokenized_sequences = self._pad_tokenized_sequences(tokenized_sequences)

    @overrides
    def get_tokenized_sequences(self) -> torch.Tensor:
        return self._tokenized_sequences

    @overrides
    def sort_batch(
            self,
            sort_tensor: torch.Tensor = None):
        perm_idx = super().sort_batch(sort_tensor)

        self._tokenized_sequences = self._tokenized_sequences[perm_idx]

    def _pad_tokenized_sequences(self, tokenized_sequences):
        lengths = [len(sequence) for sequence in tokenized_sequences]
        max_length = max(lengths)
        padded_sequences = np.zeros((self._batch_size, max_length), dtype=np.int64)
        for i, sequence_length in enumerate(lengths):
            padded_sequences[i][0:sequence_length] = tokenized_sequences[i][0:sequence_length]

    @property
    def tokenized_sequences(self) -> torch.Tensor:
        return self._tokenized_sequences
