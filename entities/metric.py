from typing import Dict, List
import numpy as np

from enums.accuracy_type import AccuracyType


class Metric:
    def __init__(self):
        self._accuracies: Dict[AccuracyType, List[float]] = {}
        self._losses: List[float] = []

    def add_accuracies(self, accuracies: Dict[AccuracyType, float]):
        for key, value in accuracies.items():
            if key not in self._accuracies.keys():
                self._accuracies[key] = []

            self._accuracies[key].append(value)
            self._accuracies[key] = self._accuracies[key][-5:]

    def get_current_accuracies(self) -> Dict[AccuracyType, float]:
        result = {}
        for key, value in self._accuracies.items():
            result[key] = np.mean(value, axis=0)

        return result

    def add_loss(self, loss_value: float):
        self._losses.append(loss_value)
        self._losses = self._losses[-5:]

    def get_current_loss(self) -> float:
        return np.mean(self._losses, axis=0)

    def initialize(self, metric, iterations_passed: int):
        self._losses = [metric.get_current_loss()]
        self._accuracies = {}
        accuracies = metric.get_current_accuracies()
        for key, value in accuracies.items():
            self._accuracies[key] = [value]

    @property
    def is_new(self) -> bool:
        return len(self._losses) == 0 and len(self._accuracies) == 0
