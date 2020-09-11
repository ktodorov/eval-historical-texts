from enums.argument_enum import ArgumentEnum

class Configuration(ArgumentEnum):
    KBert = 'kbert'
    XLNet = 'xlnet'
    MultiFit = 'multi-fit'
    SequenceToCharacter = 'sequence-to-char'
    TransformerSequence = 'transformer-sequence'
    BiLSTMCRF = 'bi-lstm-crf'
    CharacterToCharacter = 'char-to-char'
    CharacterToCharacterEncoderDecoder = 'char-to-char-encoder-decoder'
    CBOW = 'cbow'
