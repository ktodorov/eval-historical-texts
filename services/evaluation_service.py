from typing import List, Dict

import torch
from scipy import spatial

from enums.evaluation_type import EvaluationType


class EvaluationService:

    def evaluate(self, output: torch.Tensor, evaluation_types: List[EvaluationType]) -> Dict[EvaluationType, List]:
        """Evaluates the generated output based on the chosen evaluation types

        :param output: the generated output from the model
        :type output: torch.Tensor
        :param evaluation_types: list of different types of evaluations that should be performed
        :type evaluation_types: List[EvaluationType]
        :return: a dictionary with evaluation scores for every type
        :rtype: Dict[EvaluationType, List]
        """
        output_numpy = [x.cpu().detach().numpy() for x in output]

        evaluation_results = {}
        for evaluation_type in evaluation_types:
            evaluation_results[evaluation_type] = []

        # cosine distance
        if EvaluationType.CosineDistance in evaluation_types:
            cosine_distance = spatial.distance.cosine(
                output_numpy[0], output_numpy[1])

            evaluation_results[EvaluationType.CosineDistance].append(
                cosine_distance)

        # euclidean distance
        if EvaluationType.EuclideanDistance in evaluation_types:
            euclidean_distance = spatial.distance.euclidean(
                output_numpy[0], output_numpy[1])
            evaluation_results[EvaluationType.EuclideanDistance].append(
                euclidean_distance)

        return evaluation_results
