import jellyfish

class MetricsService:
    def calculate_jaccard_similarity(self, list1: list, list2: list) -> float:
        set1 = set(list1)
        set2 = set(list2)
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def calculate_normalized_levenshtein_distance(self, string1: str, string2: str) -> int:
        result = float(jellyfish.levenshtein_distance(string1, string2)) / max(len(string1), len(string2))
        return result