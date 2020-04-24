from typing import Tuple
import torch

from overrides import overrides

from entities.batch_representations.base_batch_representation import BaseBatchRepresentation


class SimpleBatchRepresentation(BaseBatchRepresentation):
    def __init__(
            self,
            device: str,
            batch_size: int,
            sequences: list):
        super().__init__(
            device,
            batch_size,
            sequences)

    @overrides
    def _create_batch(
        self,
        sequences: list,
        targets: list,
        pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return None, sequences, None

    @overrides
    def sort_batch(
            self,
            sort_tensor: torch.Tensor = None):
        pass

    @property
    def position_changes(self) -> list:
        return self._position_changes
