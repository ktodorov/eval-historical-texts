from enums.argument_enum import ArgumentEnum

class TextSequenceSplitType(ArgumentEnum):
    Segments = 'segment'
    Documents = 'document'
    MultiSegment = 'multi-segment'