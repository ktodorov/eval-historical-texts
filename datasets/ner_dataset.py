import os
import numpy as np

import torch

from typing import List
from overrides import overrides

from entities.ne_line import NELine
from entities.ne_collection import NECollection

from datasets.dataset_base import DatasetBase
from enums.run_type import RunType
from enums.language import Language
from enums.ner_type import NERType

from services.arguments.ner_arguments_service import NERArgumentsService
from services.pretrained_representations_service import PretrainedRepresentationsService
from services.process.ner_process_service import NERProcessService


class NERDataset(DatasetBase):
    def __init__(
            self,
            arguments_service: NERArgumentsService,
            pretrained_representations_service: PretrainedRepresentationsService,
            ner_process_service: NERProcessService,
            run_type: RunType):
        super().__init__()

        self._ner_process_service = ner_process_service
        self._pretrained_representations_service = pretrained_representations_service

        self._device = arguments_service.device
        self._include_pretrained = arguments_service.include_pretrained_model
        self._pretrained_model_size = self._pretrained_representations_service.get_pretrained_model_size()
        self._max_length = self._pretrained_representations_service.get_pretrained_max_length()

        self.ne_collection = ner_process_service.get_processed_data(run_type)

        print(f'Loaded {len(self.ne_collection)} items for \'{run_type}\' set')

    @overrides
    def __len__(self):
        return len(self.ne_collection)

    @overrides
    def __getitem__(self, idx):
        item: NELine = self.ne_collection[idx]
        entity_labels = self._ner_process_service.get_entity_labels(
            item)

        return (
            item.token_ids,
            entity_labels,
            item.tokens,
            item.position_changes
        )

    @overrides
    def use_collate_function(self) -> bool:
        return True

    @overrides
    def collate_function(self, sequences):
        return self._pad_and_sort_batch(sequences)

    def _pad_and_sort_batch(self, DataLoaderBatch):
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        sequences, targets, tokens, position_changes = batch_split

        lengths = [len(sequence) for sequence in sequences]

        max_length = max(lengths)

        padded_sequences = np.zeros((batch_size, max_length), dtype=np.int64) + \
            self._ner_process_service.get_entity_label(
                self._ner_process_service.PAD_TOKEN)
        padded_targets = np.zeros((batch_size, max_length), dtype=np.int64) + \
            self._ner_process_service.get_entity_label(
                self._ner_process_service.PAD_TOKEN)

        for i, sequence_length in enumerate(lengths):
            padded_sequences[i][0:sequence_length] = sequences[i][0:sequence_length]
            padded_targets[i][0:sequence_length] = targets[i][0:sequence_length]

        return (
            torch.from_numpy(padded_sequences).to(self._device),
            torch.from_numpy(padded_targets).to(self._device),
            torch.tensor(lengths, device=self._device),
            tokens,
            position_changes)