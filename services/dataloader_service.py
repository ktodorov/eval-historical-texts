from typing import Tuple

from torch.utils.data import DataLoader

from enums.run_type import RunType

from services.arguments_service_base import ArgumentsServiceBase
from services.dataset_service import DatasetService


class DataLoaderService:

    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            dataset_service: DatasetService):

        self._dataset_service = dataset_service

    def get_train_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset = self._dataset_service.get_dataset(RunType.Train)
        validation_dataset = self._dataset_service.get_dataset(RunType.Validation)

        data_loader_train: DataLoader = DataLoader(train_dataset)
        data_loader_validation: DataLoader = DataLoader(validation_dataset)

        # TODO: Add dataloader extraction

        return (data_loader_train, data_loader_validation)

    def get_test_dataloader(self) -> DataLoader:
        return None
