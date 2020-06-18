import re

from typing import List


class StringProcessService:
    def __init__(self):
        self._charmap = {
            0x201c: u'"',
            0x201d: u'"',
            0x2018: u"'",
            0x2019: u"'",
            'ﬀ': u'ff',
            'ﬁ': u'fi',
            'ﬂ': u'fl',
            'ﬃ': u'ffi',
            'ﬄ': u'ffl',
            '″': u'"',
            '′': u"'",
            '„': u'"',
            '«': u'"',
            '»': u'"'
        }

        self._number_regex = '^(((([0-9]*)(\.|,)([0-9]+))+)|([0-9]+))'

    def convert_string_unicode_symbols(self, text: str) -> str:
        result = text.translate(self._charmap)
        return result

    def convert_strings_unicode_symbols(self, texts: List[str]) -> List[str]:
        result = [self.convert_string_unicode_symbols(x) for x in texts]
        return result

    def replace_string_numbers(self, text: str) -> str:
        result = re.sub(self._number_regex, '0', text)
        return result

    def replace_strings_numbers(self, texts: List[str]) -> List[str]:
        result = [self.replace_string_numbers(x) for x in texts]
        return result

    def remove_string_characters(self, text: str, characters: List[str]) -> str:
        result = text
        for character in characters:
            result = result.replace(character, '')

        return result

    def remove_strings_characters(self, texts: List[str], characters: List[str]) -> List[str]:
        result = [self.remove_string_characters(x, characters) for x in texts]
        return result
