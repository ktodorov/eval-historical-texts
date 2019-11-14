from enums.run_type import RunType

from datasets.dataset_base import DatasetBase
from datasets.kbert_dataset import KBertDataset
from datasets.joint_kbert_dataset import JointKBertDataset

from services.arguments_service_base import ArgumentsServiceBase
from services.mask_service import MaskService


class DatasetService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            mask_service: MaskService):

        self._arguments_service = arguments_service
        self._mask_service = mask_service

    def get_dataset(self, run_type: RunType, language: str) -> DatasetBase:

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
