from typing import Dict, List
import numpy as np

from enums.metric_type import MetricType


class Metric:
    def __init__(self, amount_limit: int = 5):
        self._accuracies: Dict[MetricType, List[float]] = {}
        self._losses: List[float] = []
        self._amount_limit = amount_limit

    def add_accuracies(self, accuracies: Dict[MetricType, float]):
        for key, value in accuracies.items():
            if key not in self._accuracies.keys():
                self._accuracies[key] = []

            self._accuracies[key].append(value)
            if self._amount_limit:
                self._accuracies[key] = self._accuracies[key][-self._amount_limit:]

    def get_current_accuracies(self) -> Dict[MetricType, float]:
        result = {}
        for key, value in self._accuracies.items():
            result[key] = np.mean(value, axis=0)

        return result

    def get_accuracy_metric(self, metric_type: MetricType) -> float:
        if metric_type not in self._accuracies.keys():
            return 0

        result = np.mean(self._accuracies[metric_type], axis=0)
        return result

    def add_loss(self, loss_value: float):
        self._losses.append(loss_value)
        if self._amount_limit:
            self._losses = self._losses[-self._amount_limit:]

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
