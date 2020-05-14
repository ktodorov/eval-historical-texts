from enums.argument_enum import ArgumentEnum

class MetricType(ArgumentEnum):
    LevenshteinDistance = 'levenshtein-distance'
    JaccardSimilarity = 'jaccard-similarity'
    F1Score = 'f1-score'
    PrecisionScore = 'precision'
    RecallScore = 'recall'