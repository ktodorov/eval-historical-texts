import numpy as np

from typing import Tuple

import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizer

from enums.run_type import RunType

from services.arguments_service_base import ArgumentsServiceBase
from services.dataset_service import DatasetService
from services.tokenizer_service import TokenizerService


class DataLoaderService:

    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            dataset_service: DatasetService):

        self._dataset_service = dataset_service
        self._arguments_service = arguments_service

    def get_train_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        language = self._arguments_service.get_argument('language')

        train_dataset = self._dataset_service.get_dataset(
            RunType.Train, language)

        validation_dataset = self._dataset_service.get_dataset(
            RunType.Validation, language)

        data_loader_train: DataLoader = DataLoader(
            train_dataset,
            batch_size=self._arguments_service.get_argument('batch_size'),
            shuffle=self._arguments_service.get_argument('shuffle'))

        if train_dataset.use_collate_function():
            data_loader_train.collate_fn = train_dataset.collate_function

        data_loader_validation = None

        # TODO: Add dataloader extraction

        return (data_loader_train, data_loader_validation)

    def get_test_dataloader(self) -> DataLoader:
        return None
