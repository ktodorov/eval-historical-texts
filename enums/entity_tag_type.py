from enums.argument_enum import ArgumentEnum

class EntityTagType(ArgumentEnum):
    LiteralFine = 'literal-fine'
    LiteralCoarse = 'literal-coarse'
    MetonymicFine = 'metonymic-fine'
    MetonymicCoarse = 'metonymic-coarse'
    Component = 'component'
    Nested = 'nested'