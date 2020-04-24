import torch

from overrides import overrides

from entities.batch_representations.base_batch_representation import BaseBatchRepresentation


class NERBatchRepresentation(BaseBatchRepresentation):
    def __init__(
            self,
            device: str,
            batch_size: int,
            subword_sequences: list,
            targets: list,
            tokens: list,
            position_changes: list,
            character_sequences: list = [],
            pad_idx: int = 0):
        super().__init__(
            device=device,
            batch_size=batch_size,
            subword_sequences=subword_sequences,
            character_sequences=character_sequences,
            targets=targets,
            tokens=tokens,
            pad_idx=pad_idx)

        self._position_changes = position_changes

    @overrides
    def sort_batch(
            self,
            sort_tensor: torch.Tensor = None):
        perm_idx = super().sort_batch(sort_tensor)

        self._position_changes = [self._position_changes[i] for i in perm_idx]

    @property
    def position_changes(self) -> list:
        return self._position_changes
