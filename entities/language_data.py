from enums.language import Language
from typing import Dict, List, Tuple
from transformers import PreTrainedTokenizer
import math


class LanguageData:
    def __init__(
            self,
            ocr_inputs: List[int] = [],
            ocr_aligned: List[int] = [],
            gs_aligned: List[int] = []):

        self._ocr_inputs: List[int] = []  # ocr_inputs
        self._ocr_aligned: List[int] = ocr_aligned
        self._gs_aligned: List[int] = gs_aligned

    def add_entry(
            self,
            ocr_input_entry: str,
            ocr_aligned_entry: str,
            gs_aligned_entry: str,
            tokenizer: PreTrainedTokenizer):

        ocr_ids = tokenizer.encode(ocr_aligned_entry).ids
        gs_ids = tokenizer.encode(gs_aligned_entry).ids

        # We skip articles which contain more tokens than max_tokens
        max_tokens = 2000
        if len(ocr_ids) > max_tokens:
            return

        self._ocr_aligned.append(ocr_ids)
        self._gs_aligned.append(gs_ids)

    def get_entry(self, index: int) -> Tuple[List[int], List[int], List[int]]:
        if index > self.length:
            raise Exception(
                'Index given is higher than the total items in the language data')

        result = (
            None,  # self._ocr_inputs[index],
            self._ocr_aligned[index],
            self._gs_aligned[index]
        )

        return result

    def get_entries(self, length: int) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        if length > self.length:
            raise Exception(
                'Length given is greater than the total items in the language data')

        result = (
            None,  # self._ocr_inputs[:length],
            self._ocr_aligned[:length],
            self._gs_aligned[:length]
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
