import os
import numpy as np

from typing import Dict, List

from models.model_base import ModelBase

from entities.model_checkpoint import ModelCheckpoint

from enums.evaluation_type import EvaluationType
from enums.output_format import OutputFormat

from services.arguments_service_base import ArgumentsServiceBase
from services.dataloader_service import DataLoaderService
from services.evaluation_service import EvaluationService

from utils.dict_utils import update_dictionaries


class TestService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            dataloader_service: DataLoaderService,
            evaluation_service: EvaluationService,
            model: ModelBase):

        self._arguments_service = arguments_service
        self._evaluation_service = evaluation_service

        self._model = model.to(arguments_service.get_argument('device'))
        self._load_model()
        self._model.eval()

        self._dataloader = dataloader_service.get_test_dataloader()

    def test(self) -> bool:
        evaluation: Dict[EvaluationType, List] = {}

        targets = []
        for i, (batch, target) in enumerate(self._dataloader):
            batch = batch.unsqueeze(1).to(
                self._arguments_service.get_argument('device'))

            targets.append(target[0])
            outputs = self._model.forward(batch)

            batch_evaluation = self._evaluation_service.evaluate(
                outputs,
                self._arguments_service.get_argument('evaluation_type')
            )

            update_dictionaries(evaluation, batch_evaluation)

        self._save_evaluation(evaluation, targets)
        return True

    def _save_evaluation(self, evaluation: Dict[EvaluationType, List], targets: List[str]):
        output_format: OutputFormat = self._arguments_service.get_argument(
            'output_eval_format')

        if output_format == OutputFormat.SemEval:
            self._save_semeval_evaluation(evaluation, targets)
        else:
            print(evaluation)

    def _save_semeval_evaluation(self, evaluation: Dict[EvaluationType, List], targets: List[str]):
        output_folder = self._arguments_service.get_argument('output_folder')
        checkpoint_name = self._arguments_service.get_argument(
            'checkpoint_name')
        output_file_task1 = os.path.join(
            output_folder, f'{checkpoint_name}-task1.txt')
        output_file_task2 = os.path.join(
            output_folder, f'{checkpoint_name}-task2.txt')

        threshold = self._arguments_service.get_argument(
            'word_distance_threshold')

        task1_dict = {}
        distances = evaluation[EvaluationType.EuclideanDistance]

        for i, target_word in enumerate(targets):
            if distances[i] > threshold:
                task1_dict[target_word] = 1
            else:
                task1_dict[target_word] = 0

        abs_distances = [abs(distance) for distance in distances]
        max_args = np.argsort(abs_distances)

        with open(output_file_task1, 'w', encoding='utf-8') as task1_file:
            for word, class_prediction in task1_dict.items():
                task1_file.write(f'{word}\t{class_prediction}\n')

        with open(output_file_task2, 'w', encoding='utf-8') as task2_file:
            for i, target in enumerate(targets):
                task2_file.write(f'{target}\t{max_args[i]}\n')

        print('Output saved')

    def _load_model(self) -> ModelCheckpoint:
        checkpoints_path = self._get_checkpoints_path()
        model_checkpoint = self._model.load(checkpoints_path, 'BEST')
        return model_checkpoint

    def _get_checkpoints_path(self) -> str:
        if not self._arguments_service.get_argument('checkpoint_folder'):
            return self._arguments_service.get_argument('output_folder')

        return self._arguments_service.get_argument('checkpoint_folder')
