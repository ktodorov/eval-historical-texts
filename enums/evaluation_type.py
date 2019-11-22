from enum import Enum

class EvaluationType(Enum):
    CosineDistance = 'CosineDistance'
    EuclideanDistance = 'EuclideanDistance'

    def __str__(self):
        return self.value