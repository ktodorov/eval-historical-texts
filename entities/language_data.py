from enums.language import Language
from typing import Dict, List, Tuple
from transformers import PreTrainedTokenizer
import math

from services.vocabulary_service import VocabularyService

class LanguageData:
    def __init__(
            self,
            ocr_inputs: List[int] = [],
            ocr_aligned: List[int] = [],
            gs_aligned: List[int] = [],
            ocr_texts: List[int] = [],
            gs_texts: List[int] = []):

        self._ocr_inputs: List[int] = []  # ocr_inputs
        self._ocr_aligned: List[int] = ocr_aligned
        self._gs_aligned: List[int] = gs_aligned
        self._ocr_texts: List[int] = ocr_texts
        self._gs_texts: List[int] = gs_texts

    def add_entry(
            self,
            ocr_input_entry: str,
            ocr_aligned_entry: List[str],
            gs_aligned_entry: List[str],
            ocr_text: str,
            gs_text: str,
            tokenizer: PreTrainedTokenizer,
            vocabulary_service: VocabularyService):

        ocr_ids = [tokenizer.token_to_id(x) for x in ocr_aligned_entry]
        gs_ids = [tokenizer.token_to_id(x) for x in gs_aligned_entry]

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

    def get_entry(self, index: int) -> Tuple[List[int], List[int], List[int]]:
        if index > self.length:
            raise Exception(
                'Index given is higher than the total items in the language data')

        result = (
            None,  # self._ocr_inputs[index],
            self._ocr_aligned[index],
            self._gs_aligned[index],
            self._ocr_texts[index],
            self._gs_texts[index]
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
            self._gs_texts[:length]
        )

        return result

    @property
    def length(self) -> int:
        result = min(
            math.inf,  # len(self._ocr_inputs),
            len(self._ocr_aligned),
            len(self._gs_aligned)
        )

        return result
