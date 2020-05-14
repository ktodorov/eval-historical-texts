import os
import csv
import codecs

from typing import List, Dict, Tuple
from collections import defaultdict

from enums.language import Language
from enums.run_type import RunType
from enums.entity_tag_type import EntityTagType

from entities.ne_collection import NECollection
from entities.ne_line import NELine

from services.arguments.ner_arguments_service import NERArgumentsService
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.file_service import FileService
from services.data_service import DataService
from services.vocabulary_service import VocabularyService
from services.process.process_service_base import ProcessServiceBase


class NERProcessService(ProcessServiceBase):
    def __init__(
            self,
            arguments_service: NERArgumentsService,
            vocabulary_service: VocabularyService,
            file_service: FileService,
            tokenize_service: BaseTokenizeService,
            data_service: DataService):
        super().__init__()

        self._arguments_service = arguments_service
        self._tokenize_service = tokenize_service
        self._file_service = file_service
        self._data_service = data_service

        self._entity_tag_types = arguments_service.entity_tag_types

        self._data_version = "1.2"
        self.PAD_TOKEN = '[PAD]'
        self.START_TOKEN = '[CLS]'
        self.STOP_TOKEN = '[SEP]'

        self.pad_idx = 0
        self.start_idx = 1
        self.stop_idx = 2

        data_path = file_service.get_data_path()
        language_suffix = self.get_language_suffix(arguments_service.language)

        self._train_ne_collection = self.preprocess_data(
            os.path.join(
                data_path, f'HIPE-data-v{self._data_version}-train-{language_suffix}.tsv'),
            limit=arguments_service.train_dataset_limit_size)

        self._validation_ne_collection = self.preprocess_data(
            os.path.join(
                data_path, f'HIPE-data-v{self._data_version}-dev-{language_suffix}.tsv'),
            limit=arguments_service.validation_dataset_limit_size)

        self._entity_mappings = self._create_entity_mappings(
            self._train_ne_collection,
            self._validation_ne_collection)

        vocabulary_data = self._load_vocabulary_data(language_suffix, self._data_version)
        vocabulary_service.initialize_vocabulary_data(vocabulary_data)

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
            split_documents = self._arguments_service.split_documents

            for i, row in enumerate(reader):
                if ((split_documents and row['TOKEN'].startswith('# segment')) or
                    (not split_documents and row['TOKEN'].startswith('# document'))):
                    if len(current_sentence.tokens) > 0:
                        current_sentence.tokenize_text(
                            self._tokenize_service,
                            replace_all_numbers=self._arguments_service.replace_all_numbers,
                            expand_targets=not self._arguments_service.merge_subwords)

                        collection.add_line(current_sentence)
                        current_sentence = NELine()

                        if limit and len(collection) >= limit:
                            break
                elif row['TOKEN'].startswith('#'):
                    continue
                else:
                    current_sentence.add_data(row)

        # add last document
        if len(current_sentence.tokens) > 0:
            current_sentence.tokenize_text(
                self._tokenize_service,
                replace_all_numbers=self._arguments_service.replace_all_numbers,
                expand_targets=not self._arguments_service.merge_subwords)

            collection.add_line(current_sentence)

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

    def get_main_entities(self, entity_tag_type: EntityTagType) -> set:
        entity_mapping_keys = [
            key for key, value in self._entity_mappings[entity_tag_type].items() if value >= 4]
        entities = set([x[2:] for x in entity_mapping_keys if x[2:] != ''])
        return entities

    def _create_entity_mappings(
            self,
            train_ne_collection: NECollection,
            validation_ne_collection: NECollection) -> Dict[EntityTagType, Dict[str, int]]:

        entity_mappings = {
            entity_tag_type: None for entity_tag_type in self._entity_tag_types
        }

        for entity_tag_type in self._entity_tag_types:
            entities = train_ne_collection.get_unique_entity_tags(
                entity_tag_type)
            entities.extend(
                validation_ne_collection.get_unique_entity_tags(entity_tag_type))
            entities = list(set(entities))
            entities.sort(key=lambda x: '' if x is None else x)
            entity_mapping = {x: i+3 for i, x in enumerate(entities)}
            entity_mapping[self.PAD_TOKEN] = self.pad_idx
            entity_mapping[self.START_TOKEN] = self.start_idx
            entity_mapping[self.STOP_TOKEN] = self.stop_idx

            entity_mappings[entity_tag_type] = entity_mapping

        return entity_mappings

    def get_entity_labels(self, ne_line: NELine) -> List[int]:
        labels = {
            entity_tag_type: None for entity_tag_type in self._entity_tag_types
        }

        for entity_tag_type in self._entity_tag_types:
            current_entity_tags = ne_line.get_entity_tags(entity_tag_type)
            labels[entity_tag_type] = [
                self.get_entity_label(entity, entity_tag_type) for entity in current_entity_tags
            ]

        return labels

    def get_entity_label(self, entity_tag: str, entity_tag_type: EntityTagType) -> int:
        if entity_tag_type not in self._entity_mappings.keys():
            raise Exception('Invalid entity tag type')

        if entity_tag not in self._entity_mappings[entity_tag_type].keys():
            raise Exception('Invalid entity tag')

        return self._entity_mappings[entity_tag_type][entity_tag]

    def get_entity_by_label(self, label: int, entity_tag_type: EntityTagType) -> str:
        if entity_tag_type not in self._entity_mappings.keys():
            raise Exception('Invalid entity tag type')

        for entity, entity_label in self._entity_mappings[entity_tag_type].items():
            if label == entity_label:
                return entity

        raise Exception('Entity not found for this label')

    def get_labels_amount(self) -> Dict[EntityTagType, int]:
        result = {
            entity_tag_type: len(entity_mapping) for entity_tag_type, entity_mapping in self._entity_mappings.items()
        }

        return result

    def get_position_changes(self, idx: int) -> list:
        result = self._validation_ne_collection.lines[idx].position_changes
        return result

    def _load_vocabulary_data(self, language_suffix: str, data_version: str):
        pickles_path = self._file_service.get_pickles_path()
        filename = f'char-vocab-{language_suffix}-{data_version}'
        character_vocabulary_data = self._data_service.load_python_obj(
            pickles_path,
            filename,
            print_on_error=False)

        if character_vocabulary_data is not None:
            return character_vocabulary_data


        unique_characters = set()

        for ne_line in self._train_ne_collection.lines:
            current_unique_characters = set(
                [char for token in ne_line.tokens for char in token])
            unique_characters = unique_characters.union(
                current_unique_characters)

        for ne_line in self._validation_ne_collection.lines:
            current_unique_characters = set(
                [char for token in ne_line.tokens for char in token])
            unique_characters = unique_characters.union(
                current_unique_characters)

        unique_characters = sorted(list(unique_characters))
        unique_characters.insert(0, '[PAD]')
        unique_characters.insert(1, '[UNK]')
        unique_characters.insert(2, '[CLS]')
        unique_characters.insert(3, '[EOS]')

        int2char = dict(enumerate(unique_characters))
        char2int = {char: index for index, char in int2char.items()}
        vocabulary_data = {
            'characters-set': unique_characters,
            'int2char': int2char,
            'char2int': char2int
        }

        self._data_service.save_python_obj(
            vocabulary_data,
            pickles_path,
            filename,
            print_success=False)

        return vocabulary_data
