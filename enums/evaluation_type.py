from enums.argument_enum import ArgumentEnum

class EvaluationType(ArgumentEnum):
    CosineDistance = 'cosine-distance'
    EuclideanDistance = 'euclidean-distance'
    NamedEntityRecognitionMatch = 'ner-match'
    JaccardSimilarity = 'jaccard-similarity'
    LevenshteinEditDistanceImprovement = 'levenshtein-edit-distance-improvement'