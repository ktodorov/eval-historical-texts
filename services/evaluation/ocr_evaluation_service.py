import os
from typing import List, Dict
import csv

import torch
from scipy import spatial
import numpy as np

from entities.batch_representation import BatchRepresentation

from enums.evaluation_type import EvaluationType
from enums.metric_type import MetricType
from enums.run_type import RunType

from overrides import overrides

from services.evaluation.base_evaluation_service import BaseEvaluationService
from services.arguments.postocr_characters_arguments_service import PostOCRCharactersArgumentsService
from services.vocabulary_service import VocabularyService
from services.metrics_service import MetricsService
from services.file_service import FileService
from services.process.ocr_character_process_service import OCRCharacterProcessService


class OCREvaluationService(BaseEvaluationService):
    def __init__(
            self,
            arguments_service: PostOCRCharactersArgumentsService,
            vocabulary_service: VocabularyService,
            process_service: OCRCharacterProcessService,
            metrics_service: MetricsService,
            file_service: FileService):
        super().__init__()

        self._vocabulary_service = vocabulary_service
        self._process_service = process_service
        self._metrics_service = metrics_service
        self._file_service = file_service
        self._arguments_service = arguments_service

        (self._input_strings,
         self._input_target_strings,
         self._original_edit_distances,
         self._original_levenshtein_distance_sum,
         self._original_histogram) = (None, None, None, None, None)

        self.final_results = []

    @overrides
    def evaluate_batch(
            self,
            output: torch.Tensor,
            batch_input: BatchRepresentation,
            evaluation_types: List[EvaluationType],
            batch_index: int) -> Dict[EvaluationType, List]:
        """Evaluates the generated output based on the chosen evaluation types

        :param output: the generated output from the model
        :type output: torch.Tensor
        :param evaluation_types: list of different types of evaluations that should be performed
        :type evaluation_types: List[EvaluationType]
        :return: a dictionary with evaluation scores for every type
        :rtype: Dict[EvaluationType, List]
        """
        if self._input_strings is None:
            (self._input_strings,
             self._input_target_strings,
             self._original_edit_distances,
             self._original_levenshtein_distance_sum,
             self._original_histogram) = self._process_service.calculate_data_statistics(run_type=RunType.Test, log_summaries=False)

        _, _, predictions = output
        predictions = predictions.cpu().detach().numpy()

        targets = batch_input.targets[:, 1:].cpu().detach().numpy()
        predicted_characters = []
        target_characters = []

        for i in range(targets.shape[0]):
            indices = np.array(
                (targets[i] != self._vocabulary_service.pad_token), dtype=bool)
            predicted_characters.append(predictions[i])
            target_characters.append(targets[i][indices])

        evaluation = {}

        predicted_strings = [self._vocabulary_service.ids_to_string(
            x, exclude_special_tokens=True, cut_after_end_token=True) for x in predicted_characters]
        target_strings = [self._vocabulary_service.ids_to_string(
            x, exclude_special_tokens=True) for x in target_characters]

        if EvaluationType.JaccardSimilarity in evaluation_types:
            jaccard_score = np.mean([self._metrics_service.calculate_jaccard_similarity(
                target_strings[i], predicted_strings[i]) for i in range(len(predicted_strings))])

            evaluation[EvaluationType.JaccardSimilarity] = [jaccard_score]

        if EvaluationType.LevenshteinEditDistanceImprovement in evaluation_types:
            predicted_levenshtein_distances = [
                self._metrics_service.calculate_levenshtein_distance(
                    predicted_string, target_string) for predicted_string, target_string in zip(predicted_strings, target_strings)
            ]
            evaluation[EvaluationType.LevenshteinEditDistanceImprovement] = predicted_levenshtein_distances

        indices_order = [
            self._input_target_strings.index(target_string) for target_string in target_strings
        ]

        input_strings = [self._input_strings[i]
                         for i in indices_order]

        input_target_strings = [
            self._input_target_strings[i] for i in indices_order]

        input_distances = [
            self._original_edit_distances[i] for i in indices_order]

        for x in zip(input_strings, predicted_strings, target_strings, input_distances, predicted_levenshtein_distances):
            self.final_results.append(x)

        return evaluation

    @overrides
    def save_results(self, evaluation: Dict[EvaluationType, List]):
        edit_distances = evaluation[EvaluationType.LevenshteinEditDistanceImprovement]
        predicted_edit_sum = sum(edit_distances)

        original_edit_sum = self._original_levenshtein_distance_sum

        improvement_percentage = (
            1 - (float(predicted_edit_sum) / original_edit_sum)) * 100

        checkpoints_path = self._file_service.get_checkpoints_path()
        file_path = os.path.join(
            checkpoints_path, f'output-{self._arguments_service.checkpoint_name}.csv')
        with open(file_path, 'w', encoding='utf-8', newline='') as output_file:
            csv_writer = csv.DictWriter(output_file, fieldnames=[
                'Input',
                'Prediction',
                'Target',
                'Input edit distance',
                'Predicted edit distance',
                'Difference'])

            csv_writer.writeheader()
            for result in self.final_results:
                csv_writer.writerow({
                    'Input': result[0],
                    'Prediction': result[1],
                    'Target': result[2],
                    'Input edit distance': result[3],
                    'Predicted edit distance': result[4],
                    'Difference': (result[3] - result[4])
                })

            csv_writer.writerow({'Input': f'Improvement percentage: {improvement_percentage}'})
        print(f'Improvement percentage: {improvement_percentage}')
