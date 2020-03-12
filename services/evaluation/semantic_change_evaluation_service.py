import os
from typing import List, Dict

import numpy as np
import torch

from enums.evaluation_type import EvaluationType
from enums.threshold_calculation import ThresholdCalculation

from services.arguments.semantic_arguments_service import SemanticArgumentsService
from services.file_service import FileService
from services.evaluation.base_evaluation_service import BaseEvaluationService
from services.plot_service import PlotService
from services.metrics_service import MetricsService


class SemanticChangeEvaluationService(BaseEvaluationService):
    def __init__(
            self,
            arguments_service: SemanticArgumentsService,
            file_service: FileService,
            plot_service: PlotService,
            metrics_service: MetricsService):
        super().__init__()

        self._arguments_service = arguments_service
        self._file_service = file_service
        self._plot_service = plot_service
        self._metrics_service = metrics_service

    def evaluate_batch(self, output: List[torch.Tensor], evaluation_types: List[EvaluationType]) -> Dict[EvaluationType, List]:
        output_numpy = [x.mean(dim=1).cpu().detach().numpy() for x in output]

        evaluation_results = {}
        for evaluation_type in evaluation_types:
            evaluation_results[evaluation_type] = []

        # cosine distance
        if EvaluationType.CosineDistance in evaluation_types:
            cosine_distance = self._metrics_service.calculate_cosine_distance(
                output_numpy[0], output_numpy[1])

            evaluation_results[EvaluationType.CosineDistance].append(
                cosine_distance)

        # euclidean distance
        if EvaluationType.EuclideanDistance in evaluation_types:
            euclidean_distance = self._metrics_service.calculate_euclidean_distance(
                output_numpy[0], output_numpy[1])
            evaluation_results[EvaluationType.EuclideanDistance].append(
                euclidean_distance)

        return evaluation_results

    def save_results(self, evaluation: Dict[EvaluationType, List], targets: List[str]):
        checkpoint_folder = self._file_service.get_checkpoints_path()
        if self._arguments_service.plot_distances:
            if EvaluationType.CosineDistance in evaluation.keys():
                self._plot_service.plot_histogram(
                    evaluation[EvaluationType.CosineDistance],
                    number_of_bins=25,
                    title='Cosine distance',
                    save_path=checkpoint_folder,
                    filename=f'cosine-distance-{str(self._arguments_service.language)}')

            if EvaluationType.EuclideanDistance in evaluation.keys():
                self._plot_service.plot_histogram(
                    evaluation[EvaluationType.EuclideanDistance],
                    number_of_bins=25,
                    title='Euclidean distance',
                    save_path=checkpoint_folder,
                    filename=f'euclidean-distance-{str(self._arguments_service.language)}')

        output_file_task1 = os.path.join(
            checkpoint_folder, 'task1.txt')
        output_file_task2 = os.path.join(
            checkpoint_folder, 'task2.txt')


        if EvaluationType.CosineDistance in evaluation.keys() and EvaluationType.EuclideanDistance in evaluation.keys():
            euclidean_distances = np.array(
                evaluation[EvaluationType.EuclideanDistance]) / max(evaluation[EvaluationType.EuclideanDistance])
            cosine_distances = np.array(
                evaluation[EvaluationType.CosineDistance]) / max(evaluation[EvaluationType.CosineDistance])
            distances = 0.5 * euclidean_distances + 0.5 * cosine_distances
        elif EvaluationType.CosineDistance in evaluation.keys():
            distances = evaluation[EvaluationType.CosineDistance]
        elif EvaluationType.EuclideanDistance in evaluation.keys():
            distances = evaluation[EvaluationType.EuclideanDistance]

        if self._arguments_service.word_distance_threshold is not None:
            threshold = self._arguments_service.word_distance_threshold
        elif self._arguments_service.word_distance_threshold_calculation == ThresholdCalculation.Median:
            threshold = np.median(distances)
        else:
            threshold = np.mean(distances)

        task1_dict = {}
        for i, target_word in enumerate(targets):
            if distances[i] > threshold:
                task1_dict[target_word] = 1
            else:
                task1_dict[target_word] = 0

        abs_distances = [abs(distance) for distance in distances]
        max_args = list(np.argsort(abs_distances))

        with open(output_file_task1, 'w', encoding='utf-8') as task1_file:
            for word, class_prediction in task1_dict.items():
                task1_file.write(f'{word}\t{class_prediction}\n')

        with open(output_file_task2, 'w', encoding='utf-8') as task2_file:
            for i, target in enumerate(targets):
                task2_file.write(f'{target}\t{max_args.index(i)}\n')

        print('Output saved')
