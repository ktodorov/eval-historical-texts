import numpy as np
import torch

from typing import Tuple, List

from services.pretrained_representations_service import PretrainedRepresentationsService


class BaseBatchRepresentation:
    def __init__(
            self,
            device: str,
            batch_size: int,
            sequences: list,
            targets: list,
            tokens: list = None,
            pad_idx: int = 0):

        self._batch_size = batch_size
        self._device = device

        self._tokens = tokens

        self._lengths, self._sequences, self._targets = self._create_batch(
            sequences,
            targets,
            pad_idx=pad_idx)

    def sort_batch(
        self,
        sort_tensor: torch.Tensor = None):
        if sort_tensor is None:
            sort_tensor = self.lengths[:, 0]

        _, perm_idx = sort_tensor.sort(descending=True)

        self._sequences = self._sequences[perm_idx]
        self._targets = self._targets[perm_idx]

        if self._tokens is not None:
            self._tokens = [self._tokens[i] for i in perm_idx]

        return perm_idx

    def _create_batch(
            self,
            sequences: list,
            targets: list,
            pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences_lengths = np.array([len(sequence) for sequence in sequences])
        targets_lengths = np.array([len(target) for target in targets])

        max_length = [sequences_lengths.max(axis=0), targets_lengths.max(axis=0)]

        padded_sequences = np.zeros(
            (self._batch_size, max_length[0]), dtype=np.int64)
        padded_targets = np.zeros(
            (self._batch_size, max_length[1]), dtype=np.int64)

        if pad_idx != 0:
            padded_sequences = padded_sequences.fill(pad_idx)
            padded_targets = padded_targets.fill(pad_idx)

        for i, (sequence_length, target_length) in enumerate(zip(sequences_lengths, targets_lengths)):
            padded_sequences[i][0:sequence_length] = sequences[i][0:sequence_length]
            padded_targets[i][0:target_length] = targets[i][0:target_length]

        return (
            torch.tensor(sequences_lengths, dtype=torch.long, device=self._device),
            torch.from_numpy(padded_sequences).to(self._device),
            torch.from_numpy(padded_targets).to(self._device))

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def lengths(self) -> torch.Tensor:
        return self._lengths

    @property
    def sequences(self) -> torch.Tensor:
        return self._sequences

    @property
    def targets(self) -> torch.Tensor:
        return self._targets

    @property
    def tokens(self) -> list:
        return self._tokens