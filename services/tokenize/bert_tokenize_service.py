import os

from typing import Tuple, List

from overrides import overrides

from tokenizers import BertWordPieceTokenizer, Tokenizer

import sentencepiece as spm

from enums.configuration import Configuration
from services.arguments.pretrained_arguments_service import PretrainedArgumentsService

from services.tokenize.base_tokenize_service import BaseTokenizeService

class BERTTokenizeService(BaseTokenizeService):
    def __init__(
            self,
            arguments_service: PretrainedArgumentsService):
        super().__init__()

        pretrained_weights = arguments_service.pretrained_weights
        configuration = arguments_service.configuration

        self._arguments_service = arguments_service
        vocabulary_path = os.path.join(arguments_service.data_folder, 'vocabularies', f'{pretrained_weights}-vocab.txt')
        if not os.path.exists(vocabulary_path):
            raise Exception(f'Vocabulary not found in {vocabulary_path}')

        self._tokenizer: BertWordPieceTokenizer = BertWordPieceTokenizer(
            vocabulary_path, lowercase=False, add_special_tokens=(arguments_service.configuration != Configuration.BiLSTMCRF))

    @overrides
    def encode_tokens(self, tokens: List[str]) -> List[int]:
        result = [self._tokenizer.token_to_id(x) for x in tokens]
        return result

    @overrides
    def decode_tokens(self, character_ids: List[int]) -> List[str]:
        result = [self._tokenizer.id_to_token(
            character_id) for character_id in character_ids]
        return result

    @overrides
    def decode_string(self, character_ids: List[int]) -> List[str]:
        result = self._tokenizer.decode(character_ids)
        return result

    @overrides
    def id_to_token(self, character_id: int) -> str:
        result = self._tokenizer.id_to_token(character_id)
        return result

    @overrides
    def encode_sequence(self, sequence: str) -> Tuple[List[int], List[str], List[Tuple[int,int]], List[int]]:
        encoded_representation = self._tokenizer.encode(sequence)
        return (
            encoded_representation.ids,
            encoded_representation.tokens,
            encoded_representation.offsets,
            encoded_representation.special_tokens_mask)

    @overrides
    def encode_sequences(self, sequences: List[str]) -> List[Tuple[List[int], List[str], List[Tuple[int,int]], List[int]]]:
        encoded_representations = self._tokenizer.encode_batch(sequences)
        return [(x.ids, x.tokens, x.offsets, x.special_tokens_mask) for x in encoded_representations]

    @property
    @overrides
    def vocabulary_size(self) -> int:
        return self._tokenizer.get_vocab_size()