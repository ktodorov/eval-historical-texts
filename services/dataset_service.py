from enums.configuration import Configuration
from enums.run_type import RunType

from datasets.dataset_base import DatasetBase
from datasets.kbert_dataset import KBertDataset
from datasets.joint_dataset import JointDataset
from datasets.newseye_dataset import NewsEyeDataset
from datasets.ocr_dataset import OCRDataset
from datasets.semeval_test_dataset import SemEvalTestDataset

from services.arguments_service_base import ArgumentsServiceBase
from services.file_service import FileService
from services.mask_service import MaskService
from services.tokenizer_service import TokenizerService
from services.log_service import LogService


class DatasetService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            mask_service: MaskService,
            tokenizer_service: TokenizerService,
            file_service: FileService,
            log_service: LogService):

        self._arguments_service = arguments_service
        self._mask_service = mask_service
        self._tokenizer_service = tokenizer_service
        self._file_service = file_service
        self._log_service = log_service

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
        joint_model: bool = self._arguments_service.get_argument('joint_model')
        configuration: Configuration = self._arguments_service.get_argument(
            'configuration')

        if run_type == RunType.Test and (configuration == Configuration.KBert or configuration == Configuration.XLNet):
            return SemEvalTestDataset(
                language,
                self._arguments_service,
                self._tokenizer_service)

        if not joint_model:
            reduction_size = self._arguments_service.get_argument(
                'train_dataset_reduction_size')
            if run_type == RunType.Validation:
                reduction_size = self._arguments_service.get_argument(
                    'validation_dataset_reduction_size')

            if (configuration == Configuration.KBert or configuration == Configuration.XLNet):
                result = KBertDataset(
                    language,
                    self._arguments_service,
                    self._mask_service,
                    self._file_service,
                    self._tokenizer_service,
                    self._log_service,
                    reduction=reduction_size)

            elif configuration == Configuration.MultiFit:
                result = OCRDataset(
                    self._file_service,
                    self._tokenizer_service,
                    self._log_service,
                    self._mask_service,
                    run_type,
                    language,
                    self._arguments_service.get_argument('device'),
                    self._arguments_service.get_argument(
                        'sentence_piece_vocabulary_size'),
                    reduction_size,
                    self._arguments_service.get_argument('max_articles_length'))

        elif joint_model:
            number_of_models: int = self._arguments_service.get_argument(
                'joint_model_amount')
            sub_datasets = self._create_datasets(language, number_of_models)
            result = JointDataset(sub_datasets)
        else:
            raise Exception('Unsupported configuration')

        return result

    def _create_datasets(self, language, number_of_datasets: int):
        configuration: Configuration = self._arguments_service.get_argument(
            'configuration')

        result = []
        if configuration == Configuration.KBert or configuration == Configuration.XLNet:
            result = [KBertDataset(language, self._arguments_service, self._mask_service, self._file_service, self._tokenizer_service, self._log_service, corpus_id=i+1)
                      for i in range(number_of_datasets)]
        else:
            raise Exception('Unsupported configuration')

        return result
