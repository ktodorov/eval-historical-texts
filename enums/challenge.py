from enums.argument_enum import ArgumentEnum


class Challenge(ArgumentEnum):
    NamedEntityRecognition = 'named-entity-recognition'
    NamedEntityLinking = 'named-entity-linking'

    PostOCRCorrection = 'post-ocr-correction'
    PostOCRErrorDetection = 'post-ocr-error-detection'

    SemanticChange = 'semantic-change'
