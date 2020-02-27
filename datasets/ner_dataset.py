import os
import numpy as np

import torch

from typing import List

from entities.ne_line import NELine
from entities.ne_collection import NECollection

from datasets.dataset_base import DatasetBase
from enums.run_type import RunType
from enums.language import Language
from enums.ner_type import NERType
from preprocessing.ner_preprocessing import preprocess_data

from services.ner_arguments_service import NERArgumentsService
from services.file_service import FileService
from services.tokenizer_service import TokenizerService
from services.pretrained_representations_service import PretrainedRepresentationsService


class NERDataset(DatasetBase):
    def __init__(
            self,
            arguments_service: NERArgumentsService,
            file_service: FileService,
            tokenizer_service: TokenizerService,
            pretrained_representations_service: PretrainedRepresentationsService,
            run_type: RunType):
        super().__init__()

        self._device = arguments_service.device
        self._pretrained_representations_service = pretrained_representations_service
        self._include_pretrained = arguments_service.include_pretrained_model
        self._pretrained_model_size = self._pretrained_representations_service.get_pretrained_model_size()
        self._max_length = self._pretrained_representations_service.get_pretrained_max_length()
        self._label_type = arguments_service.label_type

        data_path = file_service.get_data_path()
        file_suffix = 'train' if run_type == RunType.Train else 'dev'
        language_suffix = self._get_language_suffix(arguments_service.language)
        file_path = os.path.join(
            data_path, f'HIPE-data-v0.9-{file_suffix}-{language_suffix}.tsv')

        self.ne_collection = preprocess_data(
            file_path,
            tokenizer_service,
            arguments_service.train_dataset_limit_size if run_type == RunType.Train else arguments_service.validation_dataset_limit_size)

        print(f'Loaded {len(self.ne_collection)} items for \'{run_type}\' set')

        self._create_entity_mappings()

    def __len__(self):
        return len(self.ne_collection)

    def __getitem__(self, idx):
        item: NELine = self.ne_collection[idx]
        coarse_entity_labels = self._get_entity_labels(item)
        pretrained_result = self._get_pretrained_representation(item.token_ids)

        return item.token_ids, coarse_entity_labels, pretrained_result

    def _get_pretrained_representation(self, token_ids: List[int]):
        if not self._include_pretrained:
            return []

        token_ids_splits = [token_ids]
        if len(token_ids) > self._max_length:
            token_ids_splits = self._split_to_chunks(
                token_ids, chunk_size=self._max_length, overlap_size=2)

        pretrained_outputs = torch.zeros(
            (len(token_ids_splits), min(self._max_length, len(token_ids)), self._pretrained_model_size)).to(self._device) * -1

        for i, token_ids_split in enumerate(token_ids_splits):
            token_ids_tensor = torch.Tensor(
                token_ids_split).unsqueeze(0).long().to(self._device)
            pretrained_output = self._pretrained_representations_service.get_pretrained_representation(
                token_ids_tensor)

            _, output_length, _ = pretrained_output.shape

            pretrained_outputs[i, :output_length, :] = pretrained_output

        pretrained_result = pretrained_outputs.view(
            -1, self._pretrained_model_size)

        return pretrained_result

    def _get_language_suffix(self, language: Language):
        if language == Language.English:
            return 'en'
        elif language == Language.French:
            return 'fr'
        elif language == Language.German:
            return 'de'
        else:
            raise Exception('Unsupported language')

    def use_collate_function(self) -> bool:
        return True

    def collate_function(self, sequences):
        return self._pad_and_sort_batch(sequences)

    def _pad_and_sort_batch(self, DataLoaderBatch):
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        sequences, targets, pretrained_representations = batch_split

        lengths = [len(sequence) for sequence in sequences]

        max_length = max(lengths)

        padded_sequences = np.zeros(
            (batch_size, max_length), dtype=np.int64) * -1
        padded_targets = np.zeros(
            (batch_size, max_length), dtype=np.int64) * -1

        padded_pretrained_representations = []
        if self._include_pretrained:
            padded_pretrained_representations = torch.zeros(
                (batch_size, max_length, self._pretrained_model_size)).to(self._device) * -1

        for i, sequence_length in enumerate(lengths):
            padded_sequences[i][0:sequence_length] = sequences[i][0:sequence_length]
            padded_targets[i][0:sequence_length] = targets[i][0:sequence_length]

            if self._include_pretrained:
                padded_pretrained_representations[i][0:
                                                     sequence_length] = pretrained_representations[i][0:sequence_length]

        return self._sort_batch(
            torch.from_numpy(padded_sequences).to(self._device),
            torch.from_numpy(padded_targets).to(self._device),
            torch.tensor(lengths, device=self._device),
            padded_pretrained_representations)

    def _sort_batch(self, batch, targets, lengths, pretrained_embeddings):
        seq_lengths, perm_idx = lengths.sort(descending=True)
        seq_tensor = batch[perm_idx]
        targets_tensor = targets[perm_idx]

        if self._include_pretrained:
            pretrained_embeddings = pretrained_embeddings[perm_idx]

        return seq_tensor, targets_tensor, seq_lengths, pretrained_embeddings

    def _create_entity_mappings(self):
        self.coarse_entity_mapping = {
            None: 0,
            'B-org': 1,
            'I-org': 2,
            'B-pers': 3,
            'I-pers': 4,
            'B-loc': 5,
            'B-time': 6,
            'I-time': 7,
            'I-loc': 8,
            'B-prod': 9,
            'I-prod': 10,
        }

        self.fine_entity_mapping = {
            None: 0,
            'B-pers.ind': 1,
            'I-pers.ind': 2,
            'B-prod.media': 3,
            'I-prod.media': 4,
            'B-loc.adm.town': 5,
            'I-loc.adm.town': 6,
            'B-loc.adm.nat': 7,
            'B-org.ent': 8,
            'I-org.ent': 9,
            'B-loc.adm.reg': 10,
            'B-org.adm': 11,
            'I-org.adm': 12,
            'B-loc.oro': 13,
            'I-loc.oro': 14,
            'B-loc.phys.hydro': 15,
            'B-time.date.abs': 16,
            'I-time.date.abs': 17,
            'I-loc.adm.reg': 18,
            'B-pers.ind.articleauthor': 19,
            'I-pers.ind.articleauthor': 20,
            'B-org.ent.pressagency': 21,
            'I-org.ent.pressagency': 22,
            'B-loc.adm.sup': 23,
            'I-loc.adm.sup': 24,
            'I-loc.phys.hydro': 25,
            'B-prod.doctr': 26,
            'I-loc.adm.nat': 27,
            'B-loc.fac': 28,
            'I-loc.fac': 29,
            'B-loc.phys.geo': 30,
            'B-loc.add.phys': 31,
            'I-loc.add.phys': 32,
            'B-pers.coll': 33,
            'I-loc.phys.geo': 34,
            'I-pers.coll': 35,
            'I-prod.doctr': 36,
            'B-loc.unk': 37
        }

    def _get_entity_labels(self, ne_line: NELine) -> List[int]:
        if self._label_type == NERType.Coarse:
            return [self.coarse_entity_mapping[entity] for entity in ne_line.ne_coarse_lit]
        elif self._label_type == NERType.Fine:
            return [self.fine_entity_mapping[entity] for entity in ne_line.ne_fine_lit]
        else:
            raise Exception('Unsupported NER type for labels')