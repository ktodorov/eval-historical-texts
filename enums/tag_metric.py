from enum import Enum


class TagMetric(Enum):
    Correct = "correct"
    Incorrect = "incorrect"
    Partial = "partial"
    Missed = "missed"
    Spurious = "spurious"
    Possible = "possible"
    Actual = "actual"
    TruePositives = "TP"
    FalsePositives = "FP"
    FalseNegatives = "FN"
    PrecisionMicro = "precision-micro"
    RecallMicro = "recall-micro"
    F1ScoreMicro = "f1-score-micro"
    PrecisionMacroDoc = "precision-macro-doc"
    RecallMacroDoc = "recall-macro-doc"
    F1ScoreMacroDoc = "f1-score-macro-doc"
    PrecisionMacroDocStd = "precision-macro-doc-std"
    RecallMacroDocStd = "recall-macro-doc-std"
    F1ScoreMacroDocStd = "f1-score-macro-doc-std"
    PrecisionMacro = "precision-macro"
    RecallMacro = "recall-macro"
    F1ScoreMacro = "f1-score-macro"
    F1ScoreMacroRecomputed = "f1-score-macro-recomputed"
