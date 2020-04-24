import os
import numpy as np

import torch

from typing import List
from overrides import overrides

from entities.ne_line import NELine
from entities.ne_collection import NECollection
from entities.batch_representation import BatchRepresentation

from datasets.dataset_base import DatasetBase
from enums.run_type import RunType
from enums.language import Language
from enums.ner_type import NERType

from services.arguments.ner_arguments_service import NERArgumentsService
from services.pretrained_representations_service import PretrainedRepresentationsService
from services.vocabulary_service import VocabularyService
from services.process.ner_process_service import NERProcessService


class NERDataset(DatasetBase):
    def __init__(
            self,
            arguments_service: NERArgumentsService,
            pretrained_representations_service: PretrainedRepresentationsService,
            vocabulary_service: VocabularyService,
            ner_process_service: NERProcessService,
            run_type: RunType):
        super().__init__()

        self._ner_process_service = ner_process_service
        self._pretrained_representations_service = pretrained_representations_service
        self._vocabulary_service = vocabulary_service

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

        filtered_tokens = [token.replace('#', '') for token in item.tokens]

        character_sequence = [self._vocabulary_service.string_to_ids(token) for token in filtered_tokens]
        token_characters = [len(x) for x in character_sequence]
        character_sequence = [char for sublist in character_sequence for char in sublist]

        return (
            item.token_ids,
            entity_labels,
            filtered_tokens,
            item.position_changes,
            item.original_length,
            character_sequence,
            token_characters
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

        sequences, targets, tokens, position_changes, original_lengths, character_sequences, token_characters_count = batch_split

        pad_idx = self._ner_process_service.get_entity_label(self._ner_process_service.PAD_TOKEN)
        batch_representation = BatchRepresentation(
            device=self._device,
            batch_size=batch_size,
            subword_sequences=sequences,
            character_sequences=character_sequences,
            subword_characters_count=token_characters_count,
            targets=targets,
            tokens=tokens,
            position_changes=position_changes,
            pad_idx=pad_idx)

        lengths_tensor = torch.tensor(original_lengths, device=self._device)
        batch_representation.sort_batch(lengths_tensor)
        return batch_representation