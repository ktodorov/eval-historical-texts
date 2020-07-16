import os
import numpy as np

from typing import Dict, List

from models.model_base import ModelBase

from entities.model_checkpoint import ModelCheckpoint

from enums.evaluation_type import EvaluationType
from enums.output_format import OutputFormat

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.dataloader_service import DataLoaderService
from services.evaluation.base_evaluation_service import BaseEvaluationService
from services.file_service import FileService

from utils.dict_utils import update_dictionaries


class TestService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            dataloader_service: DataLoaderService,
            evaluation_service: BaseEvaluationService,
            file_service: FileService,
            model: ModelBase):

        self._arguments_service = arguments_service
        self._evaluation_service = evaluation_service
        self._file_service = file_service
        self._dataloader_service = dataloader_service

        self._model = model.to(arguments_service.device)

    def test(self) -> bool:
        self._dataloader = self._dataloader_service.get_test_dataloader()
        self._load_model()
        self._model.eval()

        evaluation: Dict[EvaluationType, List] = {}
        dataloader_length = len(self._dataloader)

        for i, batch in enumerate(self._dataloader):
            print(f'{i}/{dataloader_length}         \r', end='')

            outputs = self._model.forward(batch)

            batch_evaluation = self._evaluation_service.evaluate_batch(
                outputs,
                batch,
                self._arguments_service.evaluation_type,
                i)

            update_dictionaries(evaluation, batch_evaluation)

        return self._evaluation_service.save_results(evaluation)

    def _load_model(self) -> ModelCheckpoint:
        checkpoints_path = self._file_service.get_checkpoints_path()
        model_checkpoint = self._model.load(checkpoints_path, 'BEST')
        return model_checkpoint
