from __future__ import annotations # This is so we can use Metric as type hint

from typing import Dict, List
import numpy as np

class Metric:
    def __init__(
        self,
        amount_limit: int = 5,
        metric: Metric = None):
        self._accuracies: Dict[str, List[float]] = {}
        self._losses: List[float] = []
        self._amount_limit = amount_limit

        if metric:
            self._amount_limit = metric._amount_limit
            self.initialize(metric)


    def add_accuracies(self, accuracies: Dict[str, float]):
        for key, value in accuracies.items():
            if key not in self._accuracies.keys():
                self._accuracies[key] = []

            self._accuracies[key].append(value)
            if self._amount_limit:
                self._accuracies[key] = self._accuracies[key][-self._amount_limit:]

    def get_current_accuracies(self) -> Dict[str, float]:
        result = {}
        for key, value in self._accuracies.items():
            result[key] = np.mean(value, axis=0)

        return result

    def get_accuracy_metric(self, metric_type: str) -> float:
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

    def initialize(self, metric: Metric):
        self._losses = metric._losses[-self._amount_limit:]

        self._accuracies = {}
        accuracies = metric._accuracies
        for key, value in accuracies.items():
            self._accuracies[key] = value[-self._amount_limit:]

    def contains_accuracy_metric(self, metric_key: str) -> bool:
        return metric_key in self._accuracies.keys()

    @property
    def is_new(self) -> bool:
        return len(self._losses) == 0 and len(self._accuracies) == 0
