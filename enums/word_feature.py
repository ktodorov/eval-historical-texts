from enum import Enum


class WordFeature(Enum):
    AllUpper = 0
    AllLower = 1
    IsTitle = 2
    FirstLetterUpper = 3
    FirstLetterNotUpper = 4
    Numeric = 5
    NoAlphaNumeric = 6
