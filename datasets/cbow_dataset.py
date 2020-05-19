from overrides import overrides
import random

from enums.language import Language

from entities.batch_representation import BatchRepresentation

from datasets.dataset_base import DatasetBase
from services.arguments.semantic_arguments_service import SemanticArgumentsService
from services.process.cbow_process_service import CBOWProcessService
from services.log_service import LogService

from utils import path_utils


class CBOWDataset(DatasetBase):
    def __init__(
            self,
            arguments_service: SemanticArgumentsService,
            log_service: LogService,
            cbow_process_service: CBOWProcessService,
            **kwargs):
        super().__init__()

        self._device = arguments_service.device
        self._cbow_process_service = cbow_process_service

        self._corpus_data, self._targets = cbow_process_service.load_corpus_data(arguments_service.train_dataset_limit_size)

        print(f'Loaded {len(self._corpus_data)} entries')
        log_service.log_summary(key='Entries amount',
                                value=len(self._corpus_data))

    @overrides
    def __len__(self):
        return len(self._corpus_data)

    @overrides
    def __getitem__(self, idx):
        context_words = self._corpus_data[idx]
        target = self._targets[idx]

        return context_words, target

    @overrides
    def use_collate_function(self) -> bool:
        return True

    @overrides
    def collate_function(self, sequences):
        return self._pad_and_sort_batch(sequences)

    def _pad_and_sort_batch(self, DataLoaderBatch):
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        context_word_ids, targets = batch_split
        batch_representation = BatchRepresentation(
            device=self._device,
            batch_size=batch_size,
            word_sequences=context_word_ids,
            targets=list(targets),
            pad_idx=self._cbow_process_service._pad_idx)

        batch_representation.sort_batch()

        return batch_representation