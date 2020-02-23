import os
from typing import List, Dict

from services.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService
from services.file_service import FileService


class VocabularyService:
    def __init__(
            self,
            data_service: DataService,
            file_service: FileService):

        vocabulary_data = data_service.load_python_obj(
            file_service.get_pickles_path(),
            'vocabulary')

        if not vocabulary_data:
            raise Exception('Vocabulary not found')

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
