import os

from typing import Tuple, List

from overrides import overrides

from transformers import CamembertTokenizer, CamembertConfig, XLNetTokenizer

import sentencepiece as spm

from enums.configuration import Configuration
from services.arguments.pretrained_arguments_service import PretrainedArgumentsService
from services.file_service import FileService

from services.tokenize.base_tokenize_service import BaseTokenizeService

class CamembertTokenizeService(BaseTokenizeService):
    def __init__(
            self,
            arguments_service: PretrainedArgumentsService):
        super().__init__()

        pretrained_weights = arguments_service.pretrained_weights
        configuration = arguments_service.configuration

        self._arguments_service = arguments_service

        self._tokenizer: CamembertTokenizer = CamembertTokenizer.from_pretrained(pretrained_weights)
        self._sign_tokens = [',', '.', ';']
        self._subword_prefix_symbol = '▁'

    @overrides
    def encode_tokens(self, tokens: List[str]) -> List[int]:
        result = self._tokenizer.convert_tokens_to_ids(tokens)
        return result

    @overrides
    def decode_tokens(self, character_ids: List[int]) -> List[str]:
        result = self._tokenizer.convert_ids_to_tokens(character_ids)
        return result

    @overrides
    def decode_string(self, character_ids: List[int]) -> List[str]:
        result = self._tokenizer.decode(character_ids)
        return result

    @overrides
    def id_to_token(self, character_id: int) -> str:
        result = self._tokenizer.convert_ids_to_tokens([character_id])
        return result[0]

    @overrides
    def encode_sequence(self, sequence: str) -> Tuple[List[int], List[str], List[Tuple[int,int]], List[int]]:
        tokens: List[str] = self._tokenizer.tokenize(sequence)

        offsets_result = []
        tokens_result = []
        tokens_to_encode = []
        counter = 0
        for token in tokens:
            token_str = token
            if token == self._subword_prefix_symbol:
                if counter > 0:
                    counter += 1

                continue
            elif token.startswith(self._subword_prefix_symbol):
                if counter > 0:
                    counter += 1

                token_str = token[1:]

            offsets_result.append((counter, counter + len(token_str)))
            counter += len(token_str)

            if not token.startswith(self._subword_prefix_symbol) and token not in self._sign_tokens:
                token_str = f'##{token_str}'

            tokens_result.append(token_str)
            tokens_to_encode.append(token)


        token_ids = self._tokenizer.convert_tokens_to_ids(tokens_to_encode)
        return (
            token_ids,
            tokens_result,
            offsets_result,
            None)

    @overrides
    def encode_sequences(self, sequences: List[str]) -> List[Tuple[List[int], List[str], List[Tuple[int,int]], List[int]]]:
        return [self.encode_sequence(sequence) for sequence in sequences]

    @property
    @overrides
    def vocabulary_size(self) -> int:
        return self._tokenizer.vocab_size