import os

from typing import Tuple, List

from tokenizers import BertWordPieceTokenizer, Tokenizer

import sentencepiece as spm

from enums.configuration import Configuration
from services.pretrained_arguments_service import PretrainedArgumentsService
from services.file_service import FileService


class TokenizerService:
    def __init__(
            self,
            arguments_service: PretrainedArgumentsService,
            file_service: FileService):

        pretrained_weights = arguments_service.pretrained_weights
        configuration = arguments_service.configuration

        self._file_service = file_service
        self._arguments_service = arguments_service
        self._tokenizer_loaded = True
        self._tokenizer: Tokenizer = None

        vocabulary_path = os.path.join('data', 'vocabularies', f'{pretrained_weights}-vocab.txt')
        if not os.path.exists(vocabulary_path):
            raise Exception(f'Vocabulary not found in {vocabulary_path}')

        self._tokenizer = BertWordPieceTokenizer(
            vocabulary_path, lowercase=False, add_special_tokens=(configuration != Configuration.RNNSimple))

    def load_tokenizer_model(self):
        data_path = self._file_service.get_data_path()
        tokenizer_path = os.path.join(data_path, 'tokenizer.model')
        if not os.path.exists(tokenizer_path):
            return

        self._tokenizer.Load(tokenizer_path)
        self._tokenizer_loaded = True

    def encode_tokens(self, tokens: List[str]) -> List[int]:
        result = [self.tokenizer.token_to_id(x) for x in tokens]
        return result

    def decode_tokens(self, character_ids: List[int]) -> List[str]:
        result = [self._tokenizer.id_to_token(
            character_id) for character_id in character_ids]
        return result

    def decode_string(self, character_ids: List[int]) -> str:
        result = self._tokenizer.decode(character_ids)
        return result

    def encode_sequence(self, sequence: str) -> Tuple[List[int], List[str], List[Tuple[int,int]], List[int]]:
        encoded_representation = self._tokenizer.encode(sequence)
        return (
            encoded_representation.ids,
            encoded_representation.tokens,
            encoded_representation.offsets,
            encoded_representation.special_tokens_mask)

    def encode_sequences(self, sequences: List[str]) -> List[Tuple[List[int], List[str], List[Tuple[int,int]], List[int]]]:
        encoded_representations = self._tokenizer.encode_batch(sequences)
        return [(x.ids, x.tokens, x.offsets, x.special_tokens_mask) for x in encoded_representations]

    def is_tokenizer_loaded(self) -> bool:
        return self._tokenizer_loaded

    @property
    def vocabulary_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer