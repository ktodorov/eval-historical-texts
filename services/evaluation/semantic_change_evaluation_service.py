import os
from typing import List, Dict
from overrides import overrides
import pickle

import numpy as np
import torch

from entities.batch_representation import BatchRepresentation

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

        self._target_words = []

    @overrides
    def evaluate_batch(
            self,
            output: List[torch.Tensor],
            batch_input: BatchRepresentation,
            evaluation_types: List[EvaluationType],
            batch_index: int) -> Dict[EvaluationType, List]:

        self._target_words.append(batch_input.additional_information)

        corpus1_output, corpus2_output = output
        corpus1_output = self.normalize_word_vector(corpus1_output.detach().cpu().numpy())
        corpus2_output = self.normalize_word_vector(corpus2_output.detach().cpu().numpy())

        evaluation_results = {}
        for evaluation_type in evaluation_types:
            evaluation_results[evaluation_type] = []

        # cosine distance
        if EvaluationType.CosineDistance in evaluation_types:
            cosine_distance = self._metrics_service.calculate_cosine_distance(
                corpus1_output, corpus2_output)

            evaluation_results[EvaluationType.CosineDistance].append(
                cosine_distance)

        # euclidean distance
        if EvaluationType.EuclideanDistance in evaluation_types:
            euclidean_distance = self._metrics_service.calculate_euclidean_distance(
                corpus1_output, corpus2_output)
            evaluation_results[EvaluationType.EuclideanDistance].append(
                euclidean_distance)

        return evaluation_results

    @overrides
    def save_results(self, evaluation: Dict[EvaluationType, List]):
        checkpoint_folder = self._file_service.get_checkpoints_path()

        if self._arguments_service.plot_distances:
            self._plot_distances(evaluation, checkpoint_folder)
            return

        distances_file_path = os.path.join(
            checkpoint_folder, 'distances.txt')
        output_file_task1 = os.path.join(
            checkpoint_folder, 'task1.txt')
        output_file_task2 = os.path.join(
            checkpoint_folder, 'task2.txt')

        distances = self._get_word_distances(evaluation)
        threshold = self._get_distances_threshold(distances)

        task1_dict = {}
        for i, target_word in enumerate(self._target_words):
            if distances[i] > threshold:
                task1_dict[target_word] = 1
            else:
                task1_dict[target_word] = 0

        abs_distances = [abs(distance) for distance in distances]
        max_args = list(np.argsort(abs_distances))

        with open(distances_file_path, 'w', encoding='utf-8') as distances_file:
            for target_word, distance in zip(self._target_words, distances):
                distances_file.write(f'{target_word}\t{distance}\n')

        with open(output_file_task1, 'w', encoding='utf-8') as task1_file:
            for word, class_prediction in task1_dict.items():
                task1_file.write(f'{word}\t{class_prediction}\n')

        with open(output_file_task2, 'w', encoding='utf-8') as task2_file:
            for i, target in enumerate(self._target_words):
                task2_file.write(f'{target}\t{max_args.index(i)}\n')

        print('Output saved')

    def normalize_word_vector(self, word_vec):
        norm=np.linalg.norm(word_vec)
        if norm == 0:
            return word_vec
        return word_vec/norm

    def _plot_distances(
            self,
            evaluation: Dict[EvaluationType, List],
            checkpoint_folder: str):
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

    def _get_word_distances(
            self,
            evaluation: Dict[EvaluationType, List]) -> list:
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

        return distances

    def _get_distances_threshold(
            self,
            distances: list) -> float:
        if self._arguments_service.word_distance_threshold is not None:
            threshold = self._arguments_service.word_distance_threshold
        elif self._arguments_service.word_distance_threshold_calculation == ThresholdCalculation.Median:
            threshold = np.median(distances)
        else:
            threshold = np.mean(distances)

        return threshold
