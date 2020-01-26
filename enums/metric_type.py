from enums.argument_enum import ArgumentEnum

class MetricType(ArgumentEnum):
    LevenshteinDistance = 'levenshtein-distance'
    JaccardSimilarity = 'jaccard-similarity'