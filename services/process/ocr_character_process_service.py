import os
import numpy as np
import random

from typing import List

from enums.run_type import RunType

from entities.language_data import LanguageData

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.process.process_service_base import ProcessServiceBase
from services.file_service import FileService
from services.data_service import DataService
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.metrics_service import MetricsService
from services.vocabulary_service import VocabularyService
from services.log_service import LogService
from services.download.ocr_download_service import OCRDownloadService
from services.cache_service import CacheService


class OCRCharacterProcessService(ProcessServiceBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            data_service: DataService,
            file_service: FileService,
            tokenize_service: BaseTokenizeService,
            metrics_service: MetricsService,
            vocabulary_service: VocabularyService,
            log_service: LogService,
            ocr_download_service: OCRDownloadService,
            cache_service: CacheService):

        self._arguments_service = arguments_service
        self._data_service = data_service
        self._file_service = file_service
        self._tokenize_service = tokenize_service
        self._metrics_service = metrics_service
        self._vocabulary_service = vocabulary_service
        self._log_service = log_service
        self._ocr_download_service = ocr_download_service
        self._cache_service = cache_service

        self.original_levenshtein_distance_sum: int = 0

        vocabulary_data = self._cache_service.get_item_from_cache(
            item_key='char-vocabulary',
            callback_function=self._generate_vocabulary)

        self._vocabulary_service.initialize_vocabulary_data(vocabulary_data)

    def get_language_data(self, run_type: RunType):
        language_data: LanguageData = None
        if run_type == RunType.Train:
            limit_size = self._arguments_service.train_dataset_limit_size
        elif run_type == RunType.Validation:
            limit_size = self._arguments_service.validation_dataset_limit_size
        else:
            limit_size = None

        language_data = self._load_language_data(
            run_type,
            limit_size)

        if run_type == RunType.Validation:
            (_, _, _, self.original_levenshtein_distance_sum, _) = self.calculate_data_statistics(
                language_data, log_summaries=(run_type == RunType.Validation))

        return language_data

    def calculate_data_statistics(
            self,
            language_data: LanguageData = None,
            run_type: RunType = None,
            log_summaries: bool = True):
        assert language_data is not None or run_type is not None, 'At least one of language_data or run_type must be supplied'

        if language_data is None and run_type is not None:
            language_data = self.get_language_data(run_type)

        input_strings = []
        target_strings = []
        edit_distances = []

        for idx in range(language_data.length):
            entry = language_data.get_entry(idx)
            _, _, _, ocr_text, gs_text, _ = entry

            input_string = self._vocabulary_service.ids_to_string(
                ocr_text,
                exclude_special_tokens=True)

            target_string = self._vocabulary_service.ids_to_string(
                gs_text,
                exclude_special_tokens=True)

            input_levenshtein_distance = self._metrics_service.calculate_levenshtein_distance(
                input_string,
                target_string)

            edit_distances.append(input_levenshtein_distance)

            input_strings.append(input_string)
            target_strings.append(target_string)

        original_levenshtein_distance_sum = sum(edit_distances)
        original_histogram = np.histogram(edit_distances, bins=100)

        if log_summaries:
            self._log_service.log_summary(
                'original-edit-distances-count', original_histogram[0])
            self._log_service.log_summary(
                'original-edit-distances-bins', original_histogram[1])

        return (
            input_strings,
            target_strings,
            edit_distances,
            original_levenshtein_distance_sum,
            original_histogram)

    def _generate_language_data(
            self,
            run_type: RunType):
        if run_type == RunType.Test:
            self._ocr_download_service.download_test_data(self._arguments_service.language)

            pairs = self._cache_service.get_item_from_cache(
                item_key='test-pairs',
                callback_function=self._load_eval_splits)
        else:
            self._ocr_download_service.download_training_data(self._arguments_service.language)

            train_pairs, validation_pairs = self._cache_service.get_item_from_cache(
                item_key='train-validation-pairs',
                callback_function=self._load_train_splits)

            pairs = train_pairs if run_type == RunType.Train else validation_pairs

        language_data = LanguageData.from_pairs(
            self._tokenize_service,
            self._vocabulary_service,
            pairs)

        return language_data

    def _generate_vocabulary(self):
        self._ocr_download_service.download_test_data(self._arguments_service.language)
        self._ocr_download_service.download_training_data(self._arguments_service.language)

        ocr_gs_file_data_eval_cache_key = f'ocr-gs-file-data-eval'
        ocr_gs_file_data_cache_key = f'ocr-gs-file-data'
        (ocr_file_data_eval, gs_file_data_eval) = self._cache_service.get_item_from_cache(
            item_key=ocr_gs_file_data_eval_cache_key,
            callback_function=lambda: (
                self._load_file_data(evaluation_mode=True)))

        (ocr_file_data, gs_file_data) = self._cache_service.get_item_from_cache(
            item_key=ocr_gs_file_data_cache_key,
            callback_function=lambda: (
                self._load_file_data(evaluation_mode=False)))

        full_string = ''.join(ocr_file_data_eval +
                              ocr_file_data + gs_file_data_eval + gs_file_data)
        data_characters = list(sorted(list(set(full_string))))
        data_characters.insert(0, '[PAD]')
        data_characters.insert(1, '[UNK]')
        data_characters.insert(2, '[CLS]')
        data_characters.insert(3, '[EOS]')

        # use enumeration to give the characters integer values
        int2char = dict(enumerate(data_characters))

        # create the look up dictionary from characters to the assigned integers
        char2int = {char: index for index, char in int2char.items()}

        vocabulary_data = {
            'characters-set': data_characters,
            'int2char': int2char,
            'char2int': char2int
        }

        return vocabulary_data

    def _load_language_data(
            self,
            run_type: RunType,
            reduction: int) -> LanguageData:
        language_data = self._cache_service.get_item_from_cache(
            item_key=f'language-data-{run_type.value}',
            callback_function=lambda: self._generate_language_data(run_type))

        total_amount = language_data.length
        if reduction is not None:
            language_data_items = language_data.get_entries(
                reduction)
            language_data = LanguageData(
                language_data_items[0],
                language_data_items[1],
                language_data_items[2],
                language_data_items[3],
                language_data_items[4],
                language_data_items[5],
                language_data_items[6])

        print(
            f'Loaded {language_data.length} entries out of {total_amount} total for {run_type.to_str()}')
        self._log_service.log_summary(
            key=f'\'{run_type.to_str()}\' entries amount', value=language_data.length)

        return language_data

    def _load_file_data(
            self,
            evaluation_mode: bool):
        ocr_aligned_lengths = []
        gs_aligned_lengths = []

        if evaluation_mode:
            cache_keys = ['newseye-2019-eval-dataset']
        else:
            cache_keys = [
                'trove-dataset',
                'newseye-2017-full-dataset',
                'newseye-2019-train-dataset']

        number_of_files = len(cache_keys)

        ocr_file_data = []
        gs_file_data = []

        for i, cache_key in enumerate(cache_keys):
            print(f'{i}/{number_of_files}             \r', end='')
            result = self._cache_service.get_item_from_cache(cache_key)
            if result is None:
                continue

            ocr_file_data.extend(result[0])
            gs_file_data.extend(result[1])

        return ocr_file_data, gs_file_data

    def _load_tokens_data(
            self,
            ocr_file_data: List[str],
            gs_file_data: List[str]):
        ocr_tokens = [
            self._tokenize_service.encode_sequence(ocr_file_data_obj)[0]
            for ocr_file_data_obj in ocr_file_data
        ]

        gs_tokens = [
            self._tokenize_service.encode_sequence(gs_file_data_obj)[0]
            for gs_file_data_obj in gs_file_data
        ]

        return ocr_tokens, gs_tokens

    def _read_data(self, evaluation_mode: bool):
        evaluation_mode_str = '-eval' if evaluation_mode else ''
        ocr_gs_file_data_cache_key = f'ocr-gs-file-data{evaluation_mode_str}'
        (ocr_file_data, gs_file_data) = self._cache_service.get_item_from_cache(
            item_key=ocr_gs_file_data_cache_key,
            callback_function=lambda: (
                self._load_file_data(evaluation_mode)))

        ocr_gs_tokens_cache_key = f'ocr-gs-tokens{evaluation_mode_str}'
        (ocr_tokens, gs_tokens) = self._cache_service.get_item_from_cache(
            item_key=ocr_gs_tokens_cache_key,
            callback_function=lambda: (self._load_tokens_data(ocr_file_data, gs_file_data)))

        token_pairs_cache_key = f'metrics-token-pairs{evaluation_mode_str}'
        token_pairs = self._cache_service.get_item_from_cache(
            item_key=token_pairs_cache_key,
            callback_function=lambda: (self._generate_token_pairs(ocr_tokens, gs_tokens)))

        decoded_pairs_cache_key = f'metrics-decoded-pairs{evaluation_mode_str}'
        decoded_pairs = self._cache_service.get_item_from_cache(
            item_key=decoded_pairs_cache_key,
            callback_function=lambda: (list(zip(ocr_file_data, gs_file_data))))

        return token_pairs, decoded_pairs

    def _generate_token_pairs(
            self,
            ocr_tokens: List[List[int]],
            gs_tokens: List[List[int]],):
        token_pairs = [
            ([self._tokenize_service.id_to_token(x) for x in ocr_tokens[i]],
             [self._tokenize_service.id_to_token(x) for x in gs_tokens[i]])
            for i in range(len(ocr_tokens))
        ]

        return token_pairs

    def _load_train_splits(self):
        token_pairs, decoded_pairs = self._read_data(evaluation_mode=False)
        eval_indices = random.sample(
            range(len(token_pairs)),
            int(0.01 * len(token_pairs)))

        train_pairs = []
        eval_pairs = [[token_pairs[i], decoded_pairs[i]]
                      for i in eval_indices]

        eval_indices_dict = {i: False for i in range(len(token_pairs))}
        for i in eval_indices:
            eval_indices_dict[i] = True

        train_pairs = [[token_pairs[i], decoded_pairs[i]]
                       for i in range(len(token_pairs))
                       if not eval_indices_dict[i]]

        return train_pairs, eval_pairs

    def _load_eval_splits(self):
        token_pairs, decoded_pairs = self._read_data(evaluation_mode=True)
        test_pairs = tuple(zip(token_pairs, decoded_pairs))
        return test_pairs
