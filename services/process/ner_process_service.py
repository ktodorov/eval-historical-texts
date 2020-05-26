import os
import csv
import codecs
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

from enums.language import Language
from enums.run_type import RunType
from enums.entity_tag_type import EntityTagType
from enums.text_sequence_split_type import TextSequenceSplitType

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

        if arguments_service.evaluate:
            self._test_ne_collection = self.preprocess_data(
                os.path.join(
                    data_path,
                    f'HIPE-data-v{self._data_version}-test-masked-{language_suffix}.tsv'))

        self._entity_mappings = self._create_entity_mappings(
            self._train_ne_collection,
            self._validation_ne_collection)

        vocabulary_data = self._load_vocabulary_data(
            language_suffix, self._data_version)
        vocabulary_service.initialize_vocabulary_data(vocabulary_data)

    def preprocess_data(
            self,
            file_path: str,
            limit: int = None) -> NECollection:
        if not os.path.exists(file_path):
            raise Exception('NER File not found')

        collection = NECollection()

        multi_segments_count = []
        average_segments = []

        with open(file_path, 'r', encoding='utf-8') as tsv_file:
            reader = csv.DictReader(tsv_file, dialect=csv.excel_tab, quoting=csv.QUOTE_NONE)
            current_sentence = NELine()
            split_documents = self._arguments_service.split_type == TextSequenceSplitType.Segments
            ignore_segmentation = self._arguments_service.language == Language.English and not self._arguments_service.evaluate

            for i, row in enumerate(reader):
                is_new_segment = row['TOKEN'].startswith('# segment')
                is_new_document = row['TOKEN'].startswith('# document')
                is_comment = row['TOKEN'].startswith('#')

                if is_new_segment:
                    current_sentence.start_new_segment()

                document_id = None
                if is_new_document:
                    document_id = row['TOKEN'].split('=')[-1].strip()

                    if len(current_sentence.tokens) == 0:
                        current_sentence.document_id = document_id

                if ((split_documents and (is_new_segment or (is_comment and ignore_segmentation))) or is_new_document):
                    if len(current_sentence.tokens) > 0:
                        current_sentence.tokenize_text(
                            self._tokenize_service,
                            replace_all_numbers=self._arguments_service.replace_all_numbers,
                            expand_targets=not self._arguments_service.merge_subwords)

                        if self._arguments_service.split_type == TextSequenceSplitType.MultiSegment:
                            multi_segment_documents, ms_count, s_avg = self._split_sentence_to_multi_segments(
                                current_sentence)
                            multi_segments_count.append(ms_count)
                            average_segments.append(s_avg)
                            collection.add_lines(multi_segment_documents)
                        else:
                            collection.add_line(current_sentence)

                        current_sentence = NELine()
                        if document_id is not None:
                            current_sentence.document_id = document_id

                        if limit and len(collection) >= limit:
                            break
                elif is_comment:
                    continue
                else:
                    current_sentence.add_data(row, self._entity_tag_types)

        # add last document
        if len(current_sentence.tokens) > 0:
            current_sentence.tokenize_text(
                self._tokenize_service,
                replace_all_numbers=self._arguments_service.replace_all_numbers,
                expand_targets=not self._arguments_service.merge_subwords)

            collection.add_line(current_sentence)

        if self._arguments_service.split_type == TextSequenceSplitType.MultiSegment:
            print(f'Average multi segments per document: {np.mean(multi_segments_count)}\nAverage segments per multi segment:{np.mean(average_segments)}')

        return collection

    def get_processed_data(self, run_type: RunType):
        if run_type == RunType.Train:
            return self._train_ne_collection
        elif run_type == RunType.Validation:
            return self._validation_ne_collection
        elif run_type == RunType.Test:
            if not self._arguments_service.evaluate:
                raise Exception('You must have an evaluation run to use test collection')
            return self._test_ne_collection

        raise Exception('Unsupported run type')

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

    def get_entity_names(self, entity_tag_type: EntityTagType) -> List[str]:
        if entity_tag_type not in self._entity_mappings.keys():
            raise Exception('Invalid entity tag type')

        # get the keys (entity names) for this entity tag type ordered by their values
        result = list(
            sorted(
                self._entity_mappings[entity_tag_type],
                key=self._entity_mappings[entity_tag_type].get))

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

    def _split_sentence_to_multi_segments(self, ne_line: NELine) -> List[NELine]:
        # 1 Decide how many multi-segments are we going to have (N)
        # max_tokens_length = self._arguments_service.pretrained_max_length
        max_tokens_length = 250

        # calculate segment statistics
        segment_start_positions = [i for i, is_segment_start in enumerate(
            ne_line.segment_start) if is_segment_start]
        segment_lengths = [(segment_start_positions[i+1] - segment_start_positions[i])
                           for i in range(len(segment_start_positions)-1)]
        segment_lengths.append(len(ne_line.tokens) -
                               segment_start_positions[-1])

        multi_segments = []
        current_segment = []
        current_length = 0
        for i, (segment_start_position, segment_length) in enumerate(zip(segment_start_positions, segment_lengths)):
            if current_length + segment_length > max_tokens_length:
                multi_segments.append(current_segment)
                current_segment = []
                current_length = 0

            current_length += segment_length
            current_segment.append(i)

        multi_segments.append(current_segment)

        result = []
        # Split sentence into N multi-segments
        for segment_idx, multi_segment in enumerate(multi_segments):
            segment_start_id = segment_start_positions[min(multi_segment)]
            segment_end_id = segment_start_positions[max(
                multi_segment)] + segment_lengths[max(multi_segment)]

            multi_segment_line = NELine()
            multi_segment_line.tokens = ne_line.tokens[segment_start_id: segment_end_id]
            multi_segment_line.token_ids = ne_line.token_ids[segment_start_id: segment_end_id]
            multi_segment_line.tokens_features = ne_line.tokens_features[
                segment_start_id: segment_end_id]

            # copy tag targets
            multi_segment_line.misc = self._copy_line_targets(ne_line.misc, segment_start_id, segment_end_id, ne_line.position_changes)
            multi_segment_line.ne_coarse_lit = self._copy_line_targets(ne_line.ne_coarse_lit, segment_start_id, segment_end_id, ne_line.position_changes)
            multi_segment_line.ne_coarse_meto = self._copy_line_targets(ne_line.ne_coarse_meto, segment_start_id, segment_end_id, ne_line.position_changes)
            multi_segment_line.ne_fine_lit = self._copy_line_targets(ne_line.ne_fine_lit, segment_start_id, segment_end_id, ne_line.position_changes)
            multi_segment_line.ne_fine_meto = self._copy_line_targets(ne_line.ne_fine_meto, segment_start_id, segment_end_id, ne_line.position_changes)
            multi_segment_line.ne_fine_comp = self._copy_line_targets(ne_line.ne_fine_comp, segment_start_id, segment_end_id, ne_line.position_changes)

            multi_segment_line.ne_nested = self._copy_line_targets(ne_line.ne_nested, segment_start_id, segment_end_id, ne_line.position_changes)
            multi_segment_line.position_changes = self._cut_position_changes(
                ne_line.position_changes, segment_start_id, segment_end_id)
            result.append(multi_segment_line)

            multi_segment_line.document_id = ne_line.document_id
            multi_segment_line.segment_idx = segment_idx


        return result, len(result), np.mean([len(x) for x in multi_segments])

    def _copy_line_targets(self, target_values: list, start_idx: int, end_idx: int, position_changes: Dict[int, List[int]]) -> list:
        if target_values is None or len(target_values) == 0:
            return None

        result_targets = []
        for original_position, new_positions in position_changes.items():
            if any((x >= start_idx and x < end_idx) for x in new_positions):
                result_targets.append(target_values[original_position])

        return result_targets


    def _cut_position_changes(self, item_position_changes, start_idx, end_idx):
        result = {}
        counter = 0
        for original_position, new_positions in item_position_changes.items():
            if all(position < start_idx or position >= end_idx for position in new_positions):
                continue

            result[counter] = [(x-start_idx)
                               for x in new_positions if x >= start_idx and x < end_idx]
            counter += 1

        return result

    def _select_random_line_segments(self, ne_line: NELine) -> Tuple[int, int]:
        segment_start_ids = [i for i, is_segment_start in enumerate(
            ne_line.segment_start) if is_segment_start]

        previous_segment_start_idx = None
        chosen_segment_start_id = None
        token_count = 0
        for i, segment_start_id in enumerate(reversed(segment_start_ids)):
            if previous_segment_start_idx is None:
                token_count += len(ne_line.tokens[segment_start_id:])
            else:
                token_count += len(
                    ne_line.tokens[segment_start_id:previous_segment_start_idx])

            if token_count >= self._arguments_service.pretrained_max_length:
                break

            previous_segment_start_idx = segment_start_id
            chosen_segment_start_id = len(segment_start_ids) - i

        last_possible_segment_start_idx = chosen_segment_start_id
        if last_possible_segment_start_idx is None:
            last_possible_segment_start_idx = len(segment_start_ids) - 1

        # last_possible_segment_start_idx is the segment count that can select last,
        # all segments after this will be shorter than what we want

        start_segment_id = random.randint(0, last_possible_segment_start_idx)
        end_segment_id = None
        # we want to select multi segment that has as many segments as is the max length of BERT
        multi_segment_length = 0
        previous_segment_start_idx = segment_start_ids[start_segment_id]
        for i, segment_id in enumerate(segment_start_ids[start_segment_id+1:]):
            new_length = (segment_id - previous_segment_start_idx)
            if multi_segment_length + new_length >= self._arguments_service.pretrained_max_length:
                break

            multi_segment_length += new_length
            end_segment_id = start_segment_id + i + 1
            previous_segment_start_idx = segment_id

        return segment_start_ids[start_segment_id], segment_start_ids[end_segment_id]
