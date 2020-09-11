from enums.configuration import Configuration
from enums.run_type import RunType

from datasets.dataset_base import DatasetBase
from datasets.semantic_change_dataset import SemanticChangeDataset
from datasets.cbow_dataset import CBOWDataset
from datasets.joint_dataset import JointDataset
from datasets.newseye_dataset import NewsEyeDataset
from datasets.ocr_dataset import OCRDataset
from datasets.ocr_sequence_dataset import OCRSequenceDataset
from datasets.ocr_character_dataset import OCRCharacterDataset
from datasets.semeval_test_dataset import SemEvalTestDataset
from datasets.ner_dataset import NERDataset

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.file_service import FileService
from services.mask_service import MaskService
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.log_service import LogService
from services.vocabulary_service import VocabularyService
from services.metrics_service import MetricsService
from services.data_service import DataService
from services.process.process_service_base import ProcessServiceBase


class DatasetService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            mask_service: MaskService,
            tokenize_service: BaseTokenizeService,
            file_service: FileService,
            log_service: LogService,
            metrics_service: MetricsService,
            vocabulary_service: VocabularyService,
            data_service: DataService,
            process_service: ProcessServiceBase):

        self._arguments_service = arguments_service
        self._mask_service = mask_service
        self._tokenize_service = tokenize_service
        self._file_service = file_service
        self._log_service = log_service
        self._vocabulary_service = vocabulary_service
        self._metrics_service = metrics_service
        self._data_service = data_service
        self._process_service = process_service

    def get_dataset(self, run_type: RunType, language: str) -> DatasetBase:
        """Loads and returns the dataset based on run type ``(Train, Validation, Test)`` and the language

        :param run_type: used to distinguish which dataset should be returned
        :type run_type: RunType
        :param language: language of the text that will be used
        :type language: str
        :raises Exception: if the chosen configuration is not supported, exception will be thrown
        :return: the dataset
        :rtype: DatasetBase
        """
        joint_model: bool = self._arguments_service.joint_model
        configuration: Configuration = self._arguments_service.configuration

        if run_type == RunType.Test:
            if (configuration == Configuration.KBert or
                configuration == Configuration.XLNet or
                    configuration == Configuration.CBOW):
                return SemEvalTestDataset(
                    language,
                    self._arguments_service,
                    self._tokenize_service,
                    self._file_service,
                    self._vocabulary_service)
            elif configuration == Configuration.BiLSTMCRF:
                return NERDataset(
                    self._arguments_service,
                    self._vocabulary_service,
                    self._process_service,
                    RunType.Test)
            elif (configuration == Configuration.CharacterToCharacter or configuration == Configuration.CharacterToCharacterEncoderDecoder):
                return OCRCharacterDataset(
                    self._arguments_service,
                    self._file_service,
                    self._tokenize_service,
                    self._vocabulary_service,
                    self._metrics_service,
                    self._log_service,
                    self._data_service,
                    self._process_service,
                    run_type)
        if not joint_model:
            if (configuration == Configuration.KBert or configuration == Configuration.XLNet):
                result = SemanticChangeDataset(
                    language,
                    self._arguments_service,
                    self._mask_service,
                    self._file_service,
                    self._tokenize_service,
                    self._log_service)
            elif configuration == Configuration.CBOW:
                result = CBOWDataset(
                    self._arguments_service,
                    self._log_service,
                    self._process_service)
            elif configuration == Configuration.MultiFit:
                result = OCRDataset(
                    self._arguments_service,
                    self._file_service,
                    self._tokenize_service,
                    self._vocabulary_service,
                    self._metrics_service,
                    self._log_service,
                    self._data_service,
                    run_type)
            # TODO Fix OCRSequenceDataset implementation
            # elif configuration == Configuration.SequenceToCharacter or configuration == Configuration.TransformerSequence:
            #     result = OCRSequenceDataset(
            #         self._arguments_service,
            #         self._file_service,
            #         self._tokenize_service,
            #         self._vocabulary_service,
            #         self._metrics_service,
            #         self._log_service,
            #         self._data_service,
            #         run_type)
            elif configuration == Configuration.CharacterToCharacter or configuration == Configuration.CharacterToCharacterEncoderDecoder:
                result = OCRCharacterDataset(
                    self._arguments_service,
                    self._file_service,
                    self._tokenize_service,
                    self._vocabulary_service,
                    self._metrics_service,
                    self._log_service,
                    self._data_service,
                    self._process_service,
                    run_type)
            elif configuration == Configuration.BiLSTMCRF:
                result = NERDataset(
                    self._arguments_service,
                    self._vocabulary_service,
                    self._process_service,
                    run_type)

        elif joint_model:
            number_of_models: int = self._arguments_service.joint_model_amount
            sub_datasets = self._create_datasets(language, number_of_models)
            result = JointDataset(sub_datasets)
        else:
            raise Exception('Unsupported configuration')

        return result

    def _create_datasets(self, language, number_of_datasets: int):
        configuration = self._arguments_service.configuration

        result = []
        if configuration == Configuration.KBert or configuration == Configuration.XLNet or configuration == Configuration.CBOW:
            result = [SemanticChangeDataset(language, self._arguments_service, self._mask_service, self._file_service, self._tokenize_service, self._log_service, corpus_id=i+1)
                      for i in range(number_of_datasets)]
        else:
            raise Exception('Unsupported configuration')

        return result
