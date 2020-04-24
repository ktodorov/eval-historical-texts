import os
from typing import List, Dict

import nltk
from nltk.corpus import wordnet as wn

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService
from services.file_service import FileService


class VocabularyService:
    def __init__(
            self,
            data_service: DataService,
            file_service: FileService):

        self._data_service = data_service
        self._file_service = file_service

        vocabulary_data = data_service.load_python_obj(
            file_service.get_pickles_path(),
            'vocabulary',
            print_on_error=False)

        self.initialize_vocabulary_data(vocabulary_data)

    def initialize_vocabulary_data(self, vocabulary_data):
        if vocabulary_data is None:
            return

        self._int2char: Dict[int, str] = vocabulary_data['int2char']
        self._char2int: Dict[str, int] = vocabulary_data['char2int']

    def string_to_ids(self, input: str) -> List[int]:
        result = [self._char2int[x] for x in input]
        return result

    def ids_to_string(self, input: List[int], exclude_pad_tokens: bool = True) -> str:
        result = ''.join([self._int2char[x] for x in input])

        if exclude_pad_tokens:
            result = result.replace('[PAD]', '')

        return result

    def vocabulary_size(self) -> int:
        return len(self._int2char.keys())

    def get_all_english_nouns(self, limit_amount: int = None) -> List[str]:
        pickles_path = self._file_service.get_pickles_path()
        english_nouns_path = os.path.join(pickles_path, 'english')
        filename = f'english_nouns'
        words = self._data_service.load_python_obj(
            english_nouns_path, filename)
        if words is not None:
            return words

        filepath = os.path.join(english_nouns_path, f'{filename}.txt')
        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r', encoding='utf-8') as noun_file:
            words = [x.replace('\n', '') for x in noun_file.readlines()]

        if limit_amount:
            words = words[:limit_amount]

        words = list(set(words))

        self._data_service.save_python_obj(words, english_nouns_path, filename)

        return words

    @property
    def cls_token(self) -> int:
        return self._char2int['[CLS]']

    @property
    def eos_token(self) -> int:
        return self._char2int['[EOS]']

    @property
    def unk_token(self) -> int:
        return self._char2int['[UNK]']

    @property
    def pad_token(self) -> int:
        return self._char2int['[PAD]']
