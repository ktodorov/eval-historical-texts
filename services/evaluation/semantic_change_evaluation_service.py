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
        self._evaluation_types = None

    @overrides
    def evaluate_batch(
            self,
            output: List[torch.Tensor],
            batch_input: BatchRepresentation,
            evaluation_types: List[EvaluationType],
            batch_index: int) -> Dict[EvaluationType, List]:

        self._target_words.append(batch_input.additional_information)
        if self._evaluation_types is None:
            self._evaluation_types = evaluation_types

        if isinstance(output, tuple):
            output = output[1]

        checkpoint_folder = self._file_service.get_checkpoints_path()

        evaluation_results = {}
        for evaluation_type in evaluation_types:
            evaluation_results[evaluation_type] = []

        checkpoint_targets_path = os.path.join(checkpoint_folder, f'words-{self._arguments_service.corpus}.pickle')

        words = []
        if os.path.exists(checkpoint_targets_path):
            with open(checkpoint_targets_path, 'rb') as words_file:
                words = pickle.load(words_file)

        if len(words) > len(self._target_words):
            return evaluation_results

        # output_numpy = [x.mean(dim=1).cpu().detach().numpy() for x in output]
        output_numpy = output.mean(dim=1).squeeze().cpu().detach().numpy()

        words.append(output_numpy)
        with open(checkpoint_targets_path, 'wb') as words_file:
            words = pickle.dump(words, words_file)

        return evaluation_results

    @overrides
    def save_results(self, evaluation: Dict[EvaluationType, List]):
        checkpoint_folder = self._file_service.get_checkpoints_path()

        words1path = os.path.join(checkpoint_folder, f'words-1.pickle')
        words2path = os.path.join(checkpoint_folder, f'words-2.pickle')
        if not os.path.exists(words1path) or not os.path.exists(words2path):
            return

        with open(words1path, 'rb') as words1file:
            words1 = pickle.load(words1file)

        with open(words1path, 'rb') as words2file:
            words2 = pickle.load(words2file)

        evaluation = { et: [] for et in self._evaluation_types }
        for word1, word2 in zip(words1, words2):

            # cosine distance
            if EvaluationType.CosineDistance in self._evaluation_types:
                cosine_distance = self._metrics_service.calculate_cosine_distance(
                    word1, word2)

                evaluation[EvaluationType.CosineDistance].append(
                    cosine_distance)

            # euclidean distance
            if EvaluationType.EuclideanDistance in self._evaluation_types:
                euclidean_distance = self._metrics_service.calculate_euclidean_distance(
                    word1, word2)
                evaluation[EvaluationType.EuclideanDistance].append(
                    euclidean_distance)

        if self._arguments_service.plot_distances:
            self._plot_distances(evaluation, checkpoint_folder)
            return

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

        with open(output_file_task1, 'w', encoding='utf-8') as task1_file:
            for word, class_prediction in task1_dict.items():
                task1_file.write(f'{word}\t{class_prediction}\n')

        with open(output_file_task2, 'w', encoding='utf-8') as task2_file:
            for i, target in enumerate(self._target_words):
                task2_file.write(f'{target}\t{max_args.index(i)}\n')

        print('Output saved')

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
