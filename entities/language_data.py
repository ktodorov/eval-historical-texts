from enums.language import Language
from typing import Dict, List, Tuple
from sentencepiece import SentencePieceProcessor
import math


class LanguageData:
    def __init__(
            self,
            ocr_inputs: List[List[int]] = [],
            ocr_aligned: List[List[int]] = [],
            gs_aligned: List[List[int]] = []):

        self._ocr_inputs: List[List[int]] = []  # ocr_inputs
        self._ocr_aligned: List[List[int]] = ocr_aligned
        self._gs_aligned: List[List[int]] = gs_aligned

    def add_entry(
            self,
            ocr_input_entry: str,
            ocr_aligned_entry: str,
            gs_aligned_entry: str,
            tokenizer: SentencePieceProcessor):

        # self._ocr_inputs.append(tokenizer.EncodeAsIds(ocr_input_entry))
        self._ocr_aligned.append(tokenizer.EncodeAsIds(ocr_aligned_entry))
        self._gs_aligned.append(tokenizer.EncodeAsIds(gs_aligned_entry))

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


    def trim_entries(self, length: int):
        self._ocr_inputs = [ocr_input[:length]
                            for ocr_input in self._ocr_inputs]
        self._ocr_aligned = [ocr_aligned[:length]
                             for ocr_aligned in self._ocr_aligned]
        self._gs_aligned = [gs_aligned[:length]
                            for gs_aligned in self._gs_aligned]

    @property
    def length(self) -> int:
        result = min(
            math.inf,  # len(self._ocr_inputs),
            len(self._ocr_aligned),
            len(self._gs_aligned)
        )

        return result
