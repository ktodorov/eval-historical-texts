from enum import Enum


class Language(Enum):
    Bulgarian = 0
    Czech = 1
    German = 2
    English = 3
    Spanish = 4
    Finnish = 5
    French = 6
    Dutch = 7
    Polish = 8
    Slovak = 9
    Latin = 10
    Swedish = 11

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
