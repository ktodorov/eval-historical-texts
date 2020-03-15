import numpy as np

from typing import List

from services.metrics_service import MetricsService

class WordNeighborhood:
    def __init__(
        self,
        target_word_string: str,
        target_word_embeddings: np.array,
        neighborhood_size: int = 5):
        self.target_word_string = target_word_string
        self.target_word_embeddings = target_word_embeddings
        self.closest_words: List[str] = None
        self.closest_word_embeddings: np.array = None
        self.neighborhood_size = neighborhood_size

    def calculate_word_neighbours(
        self,
        metrics_service: MetricsService,
        all_word_strings: List[str],
        all_word_embeddings: np.array):
        cosine_distances = np.array([
            metrics_service.calculate_euclidean_distance(self.target_word_embeddings, x) for x in all_word_embeddings
        ])

        sorted_indices = cosine_distances.argsort()
        indices = []

        for index in sorted_indices:
            if all_word_strings[index] != self.target_word_string:
                indices.append(index)

            if len(indices) == self.neighborhood_size:
                break

        self.closest_words = [all_word_strings[i] for i in indices]
        self.closest_word_embeddings = all_word_embeddings[indices]
