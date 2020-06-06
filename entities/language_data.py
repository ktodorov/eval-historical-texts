import pickle

from typing import Dict, List, Tuple
import math

from services.vocabulary_service import VocabularyService
from services.tokenize.base_tokenize_service import BaseTokenizeService


class LanguageData:
    def __init__(self):
        self._ocr_inputs: List[int] = []  # ocr_inputs
        self._ocr_aligned: List[int] = []
        self._gs_aligned: List[int] = []

        self._ocr_offsets: List[int] = []
        self._gs_offsets: List[int] = []

        self._ocr_texts: List[int] = []
        self._gs_texts: List[int] = []

    @classmethod
    def from_pairs(
            cls,
            tokenize_service: BaseTokenizeService,
            vocabulary_service: VocabularyService,
            pairs: list):
        new_obj = cls()

        for pair in pairs:
            new_obj.add_entry(
                None,
                pair[0][0],
                pair[0][1],
                pair[1][0],
                pair[1][1],
                tokenize_service,
                vocabulary_service)

        return new_obj

    def add_entry(
            self,
            ocr_input_entry: str,
            ocr_aligned_entry: List[str],
            gs_aligned_entry: List[str],
            ocr_text: str,
            gs_text: str,
            tokenize_service: BaseTokenizeService,
            vocabulary_service: VocabularyService):

        ocr_ids, _, ocr_offsets, _ = tokenize_service.encode_sequence(
            ocr_text)
        gs_ids, _, gs_offsets, _ = tokenize_service.encode_sequence(gs_text)

        ocr_vocab_ids = vocabulary_service.string_to_ids(ocr_text)
        ocr_vocab_ids.insert(0, vocabulary_service.cls_token)
        ocr_vocab_ids.append(vocabulary_service.eos_token)

        gs_vocab_ids = vocabulary_service.string_to_ids(gs_text)
        gs_vocab_ids.insert(0, vocabulary_service.cls_token)
        gs_vocab_ids.append(vocabulary_service.eos_token)

        self._ocr_texts.append(ocr_vocab_ids)
        self._gs_texts.append(gs_vocab_ids)

        self._ocr_aligned.append(ocr_ids)
        self._gs_aligned.append(gs_ids)

        self._ocr_offsets.append(ocr_offsets)
        self._gs_offsets.append(gs_offsets)

    def get_entry(self, index: int) -> Tuple[List[int], List[int], List[int]]:
        if index > self.length:
            raise Exception(
                'Index given is higher than the total items in the language data')

        result = (
            None,  # self._ocr_inputs[index],
            self._ocr_aligned[index],
            self._gs_aligned[index],
            self._ocr_texts[index],
            self._gs_texts[index],
            self._ocr_offsets[index],
        )

        return result

    def get_entries(self, length: int) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        if length > self.length:
            raise Exception(
                'Length given is greater than the total items in the language data')

        result = (
            None,  # self._ocr_inputs[:length],
            self._ocr_aligned[:length],
            self._gs_aligned[:length],
            self._ocr_texts[:length],
            self._gs_texts[:length],
            self._ocr_offsets[:length],
            self._gs_offsets[:length]
        )

        return result

    def load_data(self, filepath: str):
        with open(filepath, 'rb') as data_file:
            language_data: LanguageData = pickle.load(data_file)

        if not language_data:
            return

        items_length = language_data.length

        self._ocr_aligned = language_data._ocr_aligned if hasattr(
            language_data, '_ocr_aligned') else [None] * items_length
        self._gs_aligned = language_data._gs_aligned if hasattr(
            language_data, '_gs_aligned') else [None] * items_length
        self._ocr_texts = language_data._ocr_texts if hasattr(
            language_data, '_ocr_texts') else [None] * items_length
        self._gs_texts = language_data._gs_texts if hasattr(
            language_data, '_gs_texts') else [None] * items_length
        self._ocr_offsets = language_data._ocr_offsets if hasattr(
            language_data, '_ocr_offsets') else [None] * items_length
        self._gs_offsets = language_data._gs_offsets if hasattr(
            language_data, '_gs_offsets') else [None] * items_length

    @property
    def length(self) -> int:
        result = min(
            math.inf,  # len(self._ocr_inputs),
            len(self._ocr_aligned),
            len(self._gs_aligned)
        )

        return result
