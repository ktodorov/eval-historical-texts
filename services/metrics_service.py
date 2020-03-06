import jellyfish
from sklearn.metrics import f1_score
import scipy.spatial.distance as scipy_distances


class MetricsService:
    def calculate_jaccard_similarity(self, list1: list, list2: list) -> float:
        set1 = set(list1)
        set2 = set(list2)
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def calculate_normalized_levenshtein_distance(self, string1: str, string2: str) -> int:
        result = float(jellyfish.levenshtein_distance(
            string1, string2)) / max(len(string1), len(string2))
        return result

    def calculate_f1_score(self, predictions, targets):
        result = f1_score(targets, predictions, average='micro')
        return result

    def calculate_cosine_distance(self, list1: list, list2: list):
        cosine_distance = scipy_distances.cosine(list1, list2)
        return cosine_distance

    def calculate_euclidean_distance(self, list1: list, list2: list):
        euclidean_distance = scipy_distances.euclidean(list1, list2)
        return euclidean_distance
