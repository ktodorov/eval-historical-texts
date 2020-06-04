import os
import numpy as np

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

from preprocessing.ocr_preprocessing import preprocess_data
import preprocessing.ocr_download as ocr_download


class OCRCharacterProcessService(ProcessServiceBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            data_service: DataService,
            file_service: FileService,
            tokenize_service: BaseTokenizeService,
            metrics_service: MetricsService,
            vocabulary_service: VocabularyService,
            log_service: LogService):

        self._arguments_service = arguments_service
        self._data_service = data_service
        self._file_service = file_service
        self._tokenize_service = tokenize_service
        self._metrics_service = metrics_service
        self._vocabulary_service = vocabulary_service
        self._log_service = log_service

        self.original_levenshtein_distance_sum: int = 0

    def _get_language_data_path(
            self,
            run_type: RunType):
        output_data_path = self._file_service.get_data_path()
        language_data_path = os.path.join(
            output_data_path, f'{run_type.to_str()}_language_data.pickle')

        if not os.path.exists(language_data_path):
            challenge_path = self._file_service.get_challenge_path()

            if run_type == RunType.Test:
                full_data_path = os.path.join(challenge_path, 'articles-eval')
                if not os.path.exists(full_data_path):
                    os.mkdir(full_data_path)

                newseye_2019_path = os.path.join('data', 'newseye', '2019')
                ocr_download.process_newseye_files(
                    newseye_2019_path,
                    full_data_path,
                    'newseye-2019-eval',
                    self._data_service,
                    subfolder_to_use='eval')

                pickles_path = os.path.join(
                    self._file_service.get_pickles_path(),
                    'eval')

                if not os.path.exists(pickles_path):
                    os.mkdir(pickles_path)
            else:
                full_data_path = os.path.join(challenge_path, 'articles')
                if not os.path.exists(full_data_path):
                    os.mkdir(full_data_path)

                if len(os.listdir(full_data_path)) == 0:
                    newseye_path = os.path.join('data', 'newseye')
                    trove_path = os.path.join('data', 'trove')
                    ocr_download.combine_data(
                        self._data_service,
                        full_data_path,
                        newseye_path,
                        trove_path)

                pickles_path = self._file_service.get_pickles_path()

            preprocess_data(
                self._tokenize_service,
                self._metrics_service,
                self._vocabulary_service,
                self._data_service,
                pickles_path,
                full_data_path,
                output_data_path,
                split_data=(run_type != RunType.Test))

        return language_data_path

    def _load_language_data(
            self,
            language_data_path: str,
            run_type: RunType,
            reduction: int) -> LanguageData:

        language_data = LanguageData()
        language_data.load_data(language_data_path)

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

    def get_language_data(self, run_type: RunType):
        language_data: LanguageData = None
        if run_type == RunType.Train:
            limit_size = self._arguments_service.train_dataset_limit_size
        elif run_type == RunType.Validation:
            limit_size = self._arguments_service.validation_dataset_limit_size
        else:
            limit_size = None

        language_data_path = self._get_language_data_path(run_type)
        language_data = self._load_language_data(
            language_data_path,
            run_type,
            limit_size)

        if run_type != RunType.Train:
            self._calculate_data_statistics(
                language_data, log_summaries=(run_type == RunType.Validation))

        return language_data

    def _calculate_data_statistics(self, language_data, log_summaries: bool = True):
        edit_distances = []

        for idx in range(language_data.length):
            entry = language_data.get_entry(idx)
            _, _, _, ocr_text, gs_text, _ = entry

            input_string = self._vocabulary_service.ids_to_string(ocr_text)
            target_string = self._vocabulary_service.ids_to_string(gs_text)
            input_levenshtein_distance = self._metrics_service.calculate_levenshtein_distance(
                input_string,
                target_string)

            edit_distances.append(input_levenshtein_distance)

        self.original_levenshtein_distance_sum = sum(edit_distances)

        self.original_histogram = np.histogram(edit_distances, bins=100)

        if log_summaries:
            self._log_service.log_summary(
                'original-edit-distances-count', self.original_histogram[0])
            self._log_service.log_summary(
                'original-edit-distances-bins', self.original_histogram[1])
