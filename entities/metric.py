from typing import List
import numpy as np


class Metric:
    def __init__(self):
        self._accuracies: List[float] = []
        self._losses: List[float] = []

    def add_accuracy(self, accuracy_value: float):
        self._accuracies.append(accuracy_value)

    def get_current_accuracy(self) -> float:
        return np.mean(self._accuracies, axis=0)

    def add_loss(self, loss_value: float):
        self._losses.append(loss_value)

    def get_current_loss(self) -> float:
        return np.mean(self._losses, axis=0)

    def initialize(self, metric, iterations_passed: int):
        self._losses.extend([metric.get_current_loss()] * iterations_passed)
        self._losses.extend([metric.get_current_accuracy()]
                            * iterations_passed)

    @property
    def is_new(self) -> bool:
        return len(self._losses) == 0 and len(self._accuracies) == 0
