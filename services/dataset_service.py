from enums.run_type import RunType

from datasets.dataset_base import DatasetBase
from datasets.kbert_dataset import KBertDataset
from datasets.joint_kbert_dataset import JointKBertDataset
from datasets.semeval_test_dataset import SemEvalTestDataset

from services.arguments_service_base import ArgumentsServiceBase
from services.mask_service import MaskService
from services.tokenizer_service import TokenizerService


class DatasetService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            mask_service: MaskService,
            tokenizer_service: TokenizerService):

        self._arguments_service = arguments_service
        self._mask_service = mask_service
        self._tokenizer_service = tokenizer_service

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
        if run_type == RunType.Test:
            return SemEvalTestDataset(
                language,
                self._arguments_service,
                self._tokenizer_service)

        configuration = self._arguments_service.get_argument('configuration')
        if configuration == 'kbert':
            result = KBertDataset(
                language, self._arguments_service, self._mask_service)
        elif configuration == 'joint-kbert':
            result = JointKBertDataset(
                language, self._arguments_service, self._mask_service)
        else:
            raise Exception('Unsupported configuration')

        return result
