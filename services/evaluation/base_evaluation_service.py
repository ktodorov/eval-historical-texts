from typing import List, Dict

import torch
from scipy import spatial

from entities.batch_representation import BatchRepresentation

from enums.evaluation_type import EvaluationType


class BaseEvaluationService:

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
        return {}

    def save_results(self, evaluation: Dict[EvaluationType, List], targets: List[str]):
        print(evaluation)
