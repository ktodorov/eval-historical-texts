from enum import Enum

class TagMeasureAveraging(Enum):
    Weighted = 'weighted'
    Macro = 'macro'
    Micro = 'micro'