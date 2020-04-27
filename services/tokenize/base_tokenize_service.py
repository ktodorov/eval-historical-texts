import os

from typing import Tuple, List

from tokenizers import BertWordPieceTokenizer, Tokenizer

import sentencepiece as spm

from enums.configuration import Configuration
from services.arguments.pretrained_arguments_service import PretrainedArgumentsService
from services.file_service import FileService


class BaseTokenizeService:
    def __init__(self):
        pass

    def encode_tokens(self, tokens: List[str]) -> List[int]:
        pass

    def decode_tokens(self, character_ids: List[int]) -> List[str]:
        pass

    def decode_string(self, character_ids: List[int]) -> List[str]:
        pass

    def id_to_token(self, character_id: int) -> str:
        pass

    def encode_sequence(self, sequence: str) -> Tuple[List[int], List[str], List[Tuple[int,int]], List[int]]:
        pass

    def encode_sequences(self, sequences: List[str]) -> List[Tuple[List[int], List[str], List[Tuple[int,int]], List[int]]]:
        pass

    @property
    def vocabulary_size(self) -> int:
        return 0