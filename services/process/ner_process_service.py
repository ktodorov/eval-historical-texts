import os
import csv
import codecs

from typing import List, Dict, Tuple


from enums.language import Language
from enums.run_type import RunType
from enums.ner_type import NERType

from entities.ne_collection import NECollection
from entities.ne_line import NELine

from services.arguments.ner_arguments_service import NERArgumentsService
from services.tokenizer_service import TokenizerService
from services.file_service import FileService
from services.process.process_service_base import ProcessServiceBase


class NERProcessService(ProcessServiceBase):
    def __init__(
            self,
            arguments_service: NERArgumentsService,
            file_service: FileService,
            tokenizer_service: TokenizerService):
        super().__init__()

        self._tokenizer_service = tokenizer_service
        self._label_type = arguments_service.label_type

        data_path = file_service.get_data_path()
        language_suffix = self.get_language_suffix(arguments_service.language)

        self._train_ne_collection = self.preprocess_data(
            os.path.join(
                data_path, f'HIPE-data-v1.0-train-{language_suffix}.tsv'),
            limit=arguments_service.train_dataset_limit_size)

        self._validation_ne_collection = self.preprocess_data(
            os.path.join(
                data_path, f'HIPE-data-v1.0-dev-{language_suffix}.tsv'),
            limit=arguments_service.validation_dataset_limit_size)

        self._coarse_entity_mapping, self._fine_entity_mapping = self._create_entity_mappings()

    def preprocess_data(
            self,
            file_path: str,
            limit: int = None) -> NECollection:
        if not os.path.exists(file_path):
            raise Exception('NER File not found')

        collection = NECollection()

        with open(file_path, 'r', encoding='utf-8') as tsv_file:
            reader = csv.DictReader(tsv_file, dialect=csv.excel_tab)
            current_sentence = NELine()

            for i, row in enumerate(reader):
                if row['TOKEN'].startswith('#'):
                    continue

                current_sentence.add_data(row)

                if 'EndOfLine' in row['MISC']:
                    current_sentence.tokenize_text(self._tokenizer_service)
                    collection.add_line(current_sentence)
                    current_sentence = NELine()

                    if limit and len(collection) >= limit:
                        break

        return collection

    def get_processed_data(self, run_type: RunType):
        if run_type == RunType.Train:
            return self._train_ne_collection
        else:
            return self._validation_ne_collection

    def get_language_suffix(self, language: Language):
        if language == Language.English:
            return 'en'
        elif language == Language.French:
            return 'fr'
        elif language == Language.German:
            return 'de'
        else:
            raise Exception('Unsupported language')

    def _create_entity_mappings(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        coarse_typed_entities = self._train_ne_collection.get_unique_coarse_entities()
        coarse_typed_entities.extend(
            self._validation_ne_collection.get_unique_coarse_entities())
        coarse_typed_entities = list(set(coarse_typed_entities))
        coarse_typed_entities.sort(key=lambda x: '' if x is None else x)
        coarse_entity_mapping = {x: i for i,
                                 x in enumerate(coarse_typed_entities)}

        fine_typed_entities = self._train_ne_collection.get_unique_fine_entities()
        fine_typed_entities.extend(
            self._validation_ne_collection.get_unique_fine_entities())
        fine_typed_entities = list(set(fine_typed_entities))
        fine_typed_entities.sort(key=lambda x: '' if x is None else x)
        fine_entity_mapping = {x: i for i, x in enumerate(fine_typed_entities)}

        return coarse_entity_mapping, fine_entity_mapping

    def get_entity_labels(self, ne_line: NELine) -> List[int]:
        if self._label_type == NERType.Coarse:
            return [self._coarse_entity_mapping[entity] for entity in ne_line.ne_coarse_lit]
        elif self._label_type == NERType.Fine:
            return [self._fine_entity_mapping[entity] for entity in ne_line.ne_fine_lit]
        else:
            raise Exception('Unsupported NER type for labels')

    def get_entity_by_label(self, label: int) -> str:
        if self._label_type == NERType.Coarse:
            for entity, entity_label in self._coarse_entity_mapping.items():
                if label == entity_label:
                    if entity is None:
                        return 'O'

                    return entity
        elif self._label_type == NERType.Fine:
            for entity, entity_label in self._fine_entity_mapping.items():
                if label == entity_label:
                    if entity is None:
                        return 'O'

                    return entity

        raise Exception('Entity not found for this label')

    def get_labels_amount(self) -> int:
        if self._label_type == NERType.Coarse:
            return len(self._coarse_entity_mapping)
        elif self._label_type == NERType.Fine:
            return len(self._fine_entity_mapping)

    def get_position_changes(self, idx: int) -> list:
        result = self._validation_ne_collection.lines[idx].position_changes
        return result
