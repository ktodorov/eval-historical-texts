import os
import numpy as np

import random

import torch

from typing import List, Tuple
from overrides import overrides

from entities.ne_line import NELine
from entities.ne_collection import NECollection
from entities.batch_representation import BatchRepresentation

from datasets.dataset_base import DatasetBase
from enums.run_type import RunType
from enums.language import Language
from enums.text_sequence_split_type import TextSequenceSplitType

from services.arguments.ner_arguments_service import NERArgumentsService
from services.vocabulary_service import VocabularyService
from services.process.ner_process_service import NERProcessService


class NERDataset(DatasetBase):
    def __init__(
            self,
            arguments_service: NERArgumentsService,
            vocabulary_service: VocabularyService,
            ner_process_service: NERProcessService,
            run_type: RunType):
        super().__init__()

        self._ner_process_service = ner_process_service
        self._vocabulary_service = vocabulary_service
        self._arguments_service = arguments_service

        self._device = arguments_service.device
        self._include_pretrained = arguments_service.include_pretrained_model

        self.ne_collection = ner_process_service.get_processed_data(run_type)
        self._use_multi_segment_split = arguments_service.split_type == TextSequenceSplitType.MultiSegment

        self._run_type = run_type

        print(f'Loaded {len(self.ne_collection)} items for \'{run_type}\' set')

    @overrides
    def __len__(self):
        return len(self.ne_collection)

    @overrides
    def __getitem__(self, idx):
        item: NELine = self.ne_collection[idx]

        entity_labels = []
        if self._run_type != RunType.Test:
            entity_labels = self._ner_process_service.get_entity_labels(
                item)

        filtered_tokens = [token.replace('#', '') for token in item.tokens]
        character_sequence = [self._vocabulary_service.string_to_ids(
            token) for token in filtered_tokens]
        token_characters = [len(x) for x in character_sequence]

        features = [
            [x + 1 for x in list(feature_set.values())]
            for feature_set in item.tokens_features
        ]

        return (
            item.token_ids,
            entity_labels,
            filtered_tokens,
            item.position_changes,
            character_sequence,
            token_characters,
            features,
            item.document_id,
            item.segment_idx)

    @overrides
    def use_collate_function(self) -> bool:
        return True

    @overrides
    def collate_function(self, sequences):
        return self._pad_and_sort_batch(sequences)

    def _pad_and_sort_batch(self, DataLoaderBatch):
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        (sequences,
         targets,
         tokens,
         position_changes,
         character_sequences,
         token_characters_count,
         feature_set,
         document_ids,
         segment_ids) = batch_split

        if self._run_type == RunType.Test:
            targets = None

        pad_idx = self._ner_process_service.pad_idx
        batch_representation = BatchRepresentation(
            device=self._device,
            batch_size=batch_size,
            subword_sequences=sequences,
            character_sequences=character_sequences,
            subword_characters_count=token_characters_count,
            targets=targets,
            tokens=tokens,
            position_changes=position_changes,
            manual_features=feature_set,
            additional_information=(document_ids, segment_ids),
            pad_idx=pad_idx)

        batch_representation.sort_batch()

        return batch_representation
