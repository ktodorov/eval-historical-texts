import jellyfish
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
import scipy.spatial.distance as scipy_distances

from typing import Tuple

from enums.tag_measure_type import TagMeasureType
from enums.tag_measure_averaging import TagMeasureAveraging


class MetricsService:
    def calculate_jaccard_similarity(self, list1: list, list2: list) -> float:
        if len(list1) == 0 and len(list2) == 0:
            return 0

        set1 = set(list1)
        set2 = set(list2)
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def calculate_normalized_levenshtein_distance(self, string1: str, string2: str) -> int:
        result = float(self.calculate_levenshtein_distance(
            string1, string2)) / max(len(string1), len(string2))

        return result

    def calculate_levenshtein_distance(self, string1: str, string2: str) -> int:
        result = jellyfish.levenshtein_distance(string1, string2)
        return result

    def calculate_f1_score(
            self,
            predictions,
            targets,
            tag_measure_type: TagMeasureType = TagMeasureType.Strict,
            tag_measure_averaging: TagMeasureAveraging = TagMeasureAveraging.Weighted) -> float:
        result = f1_score(targets, predictions,
                          average=tag_measure_averaging.value)
        return result

    def calculate_precision_score(
            self,
            predictions,
            targets,
            tag_measure_type: TagMeasureType = TagMeasureType.Strict,
            tag_measure_averaging: TagMeasureAveraging = TagMeasureAveraging.Weighted) -> float:
        result = precision_score(targets, predictions,
                                 average=tag_measure_averaging.value)
        return result

    def calculate_recall_score(
            self,
            predictions,
            targets,
            tag_measure_type: TagMeasureType = TagMeasureType.Strict,
            tag_measure_averaging: TagMeasureAveraging = TagMeasureAveraging.Weighted) -> float:
        result = recall_score(targets, predictions,
                              average=tag_measure_averaging.value)
        return result

    def calculate_precision_recall_fscore_support(
            self,
            predictions,
            targets,
            tag_measure_type: TagMeasureType = TagMeasureType.Strict,
            tag_measure_averaging: TagMeasureAveraging = TagMeasureAveraging.Weighted) -> Tuple[float, float, float, float]:
        result = precision_recall_fscore_support(
            targets,
            predictions,
            average=tag_measure_averaging.value,
            warn_for=tuple())

        return result

    def calculate_cosine_distance(self, list1: list, list2: list) -> float:
        cosine_distance = scipy_distances.cosine(list1, list2)
        return cosine_distance

    def calculate_euclidean_distance(self, list1: list, list2: list) -> float:
        euclidean_distance = scipy_distances.euclidean(list1, list2)
        return euclidean_distance
