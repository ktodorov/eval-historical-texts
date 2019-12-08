from enums.language import Language
from typing import Dict, List, Tuple
from sentencepiece import SentencePieceProcessor


class LanguageData:
    def __init__(self):
        self._ocr_inputs: List[List[int]] = []
        self._ocr_aligned: List[List[int]] = []
        self._gs_aligned: List[List[int]] = []

    def add_entry(
            self,
            ocr_input_entry: str,
            ocr_aligned_entry: str,
            gs_aligned_entry: str,
            tokenizer: SentencePieceProcessor):

        self._ocr_inputs.append(tokenizer.EncodeAsIds(ocr_input_entry))
        self._ocr_aligned.append(tokenizer.EncodeAsIds(ocr_aligned_entry))
        self._gs_aligned.append(tokenizer.EncodeAsIds(gs_aligned_entry))

    def get_entry(self, index: int) -> Tuple[List[int], List[int], List[int]]:
        if index > self.length:
            raise Exception(
                'Index given is higher than the total items in the language data')

        result = (
            self._ocr_inputs[index],
            self._ocr_aligned[index],
            self._gs_aligned[index]
        )

        return result

    @property
    def length(self) -> int:
        result = min(
            len(self._ocr_inputs),
            len(self._ocr_aligned),
            len(self._gs_aligned)
        )

        return result