from enum import Enum


class RunType(Enum):
    Train = 0
    Validation = 1
    Test = 2

    def to_str(self):
        if self == RunType.Train:
            return 'train'
        elif self == RunType.Validation:
            return 'validation'
        elif self == RunType.Test:
            return 'test'
        else:
            raise NotImplementedError()