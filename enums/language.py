from enums.argument_enum import ArgumentEnum


class Language(ArgumentEnum):
    Bulgarian = 'bulgarian'
    Czech = 'czech'
    German = 'german'
    English = 'english'
    Spanish = 'spanish'
    Finnish = 'finnish'
    French = 'french'
    Dutch = 'dutch'
    Polish = 'polish'
    Slovak = 'slovak'
    Latin = 'latin'
    Swedish = 'swedish'

    @staticmethod
    def from_str(language_string: str):
        if language_string.startswith('BG'):
            return Language.Bulgarian
        elif language_string.startswith('CZ'):
            return Language.Czech
        elif language_string.startswith('DE'):
            return Language.German
        elif language_string.startswith('EN'):
            return Language.English
        elif language_string.startswith('ES'):
            return Language.Spanish
        elif language_string.startswith('FI'):
            return Language.Finnish
        elif language_string.startswith('FR'):
            return Language.French
        elif language_string.startswith('NL'):
            return Language.Dutch
        elif language_string.startswith('PL'):
            return Language.Polish
        elif language_string.startswith('SL'):
            return Language.Slovak
        elif language_string.startswith('LA'):
            return Language.Latin
        elif language_string.startswith('SW'):
            return Language.Swedish

        raise Exception(
            f'Language string "{language_string}" is not supported')
