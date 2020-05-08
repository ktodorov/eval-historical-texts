import numpy as np

from typing import Tuple

import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizer

from enums.run_type import RunType

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.dataset_service import DatasetService
from services.tokenize.base_tokenize_service import BaseTokenizeService

class DataLoaderService:

    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            dataset_service: DatasetService):

        self._dataset_service = dataset_service
        self._arguments_service = arguments_service

    def get_train_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Loads and returns train and validation(if available) dataloaders

        :return: the dataloaders
        :rtype: Tuple[DataLoader, DataLoader]
        """

        language = self._arguments_service.language

        train_dataset = self._dataset_service.get_dataset(
            RunType.Train, language)

        data_loader_train: DataLoader = DataLoader(
            train_dataset,
            batch_size=self._arguments_service.batch_size,
            shuffle=self._arguments_service.shuffle)

        if train_dataset.use_collate_function():
            data_loader_train.collate_fn = train_dataset.collate_function

        if not self._arguments_service.skip_validation:
            validation_dataset = self._dataset_service.get_dataset(
                RunType.Validation, language)

            data_loader_validation = DataLoader(
                validation_dataset,
                batch_size=self._arguments_service.batch_size,
                shuffle=False)

            if validation_dataset.use_collate_function():
                data_loader_validation.collate_fn = validation_dataset.collate_function
        else:
            data_loader_validation = None

        return (data_loader_train, data_loader_validation)

    def get_test_dataloader(self) -> DataLoader:
        """Loads and returns the test dataloader

        :return: the test dataloader
        :rtype: DataLoader
        """
        language = self._arguments_service.language

        test_dataset = self._dataset_service.get_dataset(
            RunType.Test, language)

        data_loader_test: DataLoader = DataLoader(
            test_dataset,
            batch_size=self._arguments_service.batch_size,
            shuffle=False)

        if test_dataset.use_collate_function():
            data_loader_test.collate_fn = test_dataset.collate_function

        return data_loader_test
