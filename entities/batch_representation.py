import numpy as np
import torch

from typing import Tuple, List, Dict

class BatchRepresentation:
    def __init__(
            self,
            device: str,
            batch_size: int,
            character_sequences: list = [],
            subword_sequences: list = [],
            subword_characters_count: List[List[int]] = None,
            word_sequences: list = [],
            word_characters_count: List[List[int]] = None,
            targets: list = [],
            tokens: list = None,
            position_changes: Dict[int, Tuple] = None,
            offset_lists: List[Tuple] = None,
            pad_idx: int = 0,
            additional_information: object = None,
            manual_features: list = []):

        self._batch_size = batch_size
        self._device = device

        self._character_sequences, self._character_lengths = self._pad_and_convert_to_tensor(character_sequences, pad_idx)
        self._subword_sequences, self._subword_lengths = self._pad_and_convert_to_tensor(subword_sequences, pad_idx)
        self._word_sequences, self._word_lengths = self._pad_and_convert_to_tensor(word_sequences, pad_idx)

        # If we have multi-task learning we have different targets for the same sequence
        self._multi_task_learning = targets is not None and len(targets) > 0 and isinstance(targets[0], dict)
        if self._multi_task_learning:
            self._targets = {}
            self._target_lengths = {}

            target_keys = targets[0].keys()
            converted_targets = {}
            for target_key in target_keys:
                converted_targets[target_key] = [target[target_key] for target in targets]

            for key, value in converted_targets.items():
                self._targets[key], self._target_lengths[key] = self._pad_and_convert_to_tensor(value, pad_idx)
        else:
            self._targets, self._target_lengths = self._pad_and_convert_to_tensor(targets, pad_idx)

        self._subword_characters_count, _ = self._pad_and_convert_to_tensor(subword_characters_count, pad_idx)
        self._word_characters_count = word_characters_count

        self._manual_features, _ = self._pad_and_convert_to_tensor(manual_features, pad_idx)

        self._tokens = tokens
        self._offset_lists = offset_lists
        self._position_changes = position_changes
        self._additional_information = additional_information
        self._pad_idx = pad_idx

    def sort_batch(
        self,
        sort_tensor: torch.Tensor = None):
        if sort_tensor is None:
            sort_tensor = self._character_lengths
            if sort_tensor is None:
                sort_tensor = self._subword_lengths
                if sort_tensor is None:
                    sort_tensor = self._word_lengths

        _, perm_idx = sort_tensor.sort(descending=True)

        self._character_sequences = self._sort_tensor(self._character_sequences, perm_idx)
        self._character_lengths = self._sort_tensor(self._character_lengths, perm_idx)

        self._subword_sequences = self._sort_tensor(self._subword_sequences, perm_idx)
        self._subword_lengths = self._sort_tensor(self._subword_lengths, perm_idx)

        self._word_sequences = self._sort_tensor(self._word_sequences, perm_idx)
        self._word_lengths = self._sort_tensor(self._word_lengths, perm_idx)

        if self._multi_task_learning:
            self._targets = { key: self._sort_tensor(value, perm_idx) for key, value in self._targets.items() }
            self._target_lengths = { key: self._sort_tensor(value, perm_idx) for key, value in self._target_lengths.items() }
        else:
            self._targets = self._sort_tensor(self._targets, perm_idx)
            self._target_lengths = self._sort_tensor(self._target_lengths, perm_idx)

        self._subword_characters_count = self._sort_tensor(self._subword_characters_count, perm_idx)

        self._tokens = self._sort_list(self._tokens, perm_idx)
        self._offset_lists = self._sort_list(self._offset_lists, perm_idx)
        self._position_changes = self._sort_list(self._position_changes, perm_idx)

        self._word_characters_count = self._sort_list(self._word_characters_count, perm_idx)

        self._manual_features = self._sort_tensor(self._manual_features, perm_idx)

        self._additional_information = self._sort_list(self._additional_information, perm_idx)

        return perm_idx

    def generate_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        mask = (tensor != self._pad_idx).unsqueeze(-2)
        return mask

    def _sort_tensor(self, tensor: torch.Tensor, perm_idx) -> torch.Tensor:
        if tensor is None:
            return None

        return tensor[perm_idx]

    def _sort_list(self, list_to_sort: list, perm_idx) -> list:
        if list_to_sort is None:
            return None

        return [list_to_sort[i] for i in perm_idx]

    def _pad_and_convert_to_tensor(
        self,
        list_to_modify: list,
        pad_idx: int,
        return_tensor: bool = True):
        if isinstance(list_to_modify, torch.Tensor):
            return list_to_modify, None

        if list_to_modify is None or len(list_to_modify) == 0:
            return (None, None)

        batch_size = len(list_to_modify)

        use_2d_padding = isinstance(list_to_modify[0], list)
        if not use_2d_padding:
            if return_tensor:
                result_tensor = torch.tensor(list_to_modify, dtype=torch.int64, device=self._device)
                lengths_tensor = torch.ones((batch_size), dtype=torch.long, device=self._device)
                return (result_tensor, lengths_tensor)
            else:
                return (list_to_modify, np.ones((batch_size), dtype=np.long))

        use_3d_padding = isinstance(list_to_modify[0][0], list)
        lengths = np.array([len(x) for x in list_to_modify])
        max_length = max(lengths)

        if use_3d_padding:
            sub_max_length = max([max([len(y) for y in x]) for x in list_to_modify])
            padded_list = np.zeros((batch_size, max_length, sub_max_length), dtype=np.int64)
        else:
            padded_list = np.zeros((batch_size, max_length), dtype=np.int64)

        if pad_idx != 0:
            padded_list.fill(pad_idx)

        for i, length in enumerate(lengths):
            if use_3d_padding:
                padded_sublist, _ = self._pad_and_convert_to_tensor(list_to_modify[i], pad_idx, return_tensor=False)
                max_sublength = padded_sublist.shape[-1]
                padded_list[i, :length, :max_sublength] = padded_sublist
            else:
                padded_list[i, :length] = list_to_modify[i]

        if return_tensor:
            return (
                torch.from_numpy(padded_list).to(self._device),
                torch.tensor(lengths, dtype=torch.long, device=self._device))
        else:
            return (padded_list, lengths)

    def __str__(self):
        return f'batch-representation (size: {self._batch_size}, device: {self._device})'

    def __repr__(self):
        return str(self)

    def __unicode__(self):
        return str(self)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def character_lengths(self) -> torch.Tensor:
        return self._character_lengths

    @property
    def character_sequences(self) -> torch.Tensor:
        return self._character_sequences

    @property
    def subword_lengths(self) -> torch.Tensor:
        return self._subword_lengths

    @property
    def subword_sequences(self) -> torch.Tensor:
        return self._subword_sequences

    @property
    def word_lengths(self) -> torch.Tensor:
        return self._word_lengths

    @property
    def word_sequences(self) -> torch.Tensor:
        return self._word_sequences

    @property
    def targets(self) -> torch.Tensor:
        return self._targets

    @property
    def target_lengths(self) -> torch.Tensor:
        return self._target_lengths

    @property
    def manual_features(self) -> torch.Tensor:
        return self._manual_features

    @property
    def tokens(self) -> list:
        return self._tokens

    @property
    def offset_lists(self) -> list:
        return self._offset_lists

    @property
    def position_changes(self) -> list:
        return self._position_changes

    @property
    def subword_characters_count(self) -> List[List[int]]:
        return self._subword_characters_count

    @property
    def word_characters_count(self) -> List[List[int]]:
        return self._word_characters_count

    @property
    def additional_information(self) -> object:
        return self._additional_information