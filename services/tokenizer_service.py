import os

from transformers import PreTrainedTokenizer, BertTokenizer, XLNetTokenizer

import sentencepiece as spm

from enums.configuration import Configuration
from services.arguments_service_base import ArgumentsServiceBase
from services.file_service import FileService


class TokenizerService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            file_service: FileService):

        pretrained_weights = arguments_service.get_argument(
            'pretrained_weights')
        configuration: Configuration = arguments_service.get_argument(
            'configuration')

        self._file_service = file_service
        self._arguments_service = arguments_service
        self._tokenizer_loaded = True

        if configuration == Configuration.KBert:
            self._tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        elif configuration == Configuration.XLNet:
            self._tokenizer = XLNetTokenizer.from_pretrained(
                pretrained_weights)
        elif configuration == Configuration.MultiFit:
            self._tokenizer_loaded = False
            self._tokenizer = spm.SentencePieceProcessor()
            self.load_tokenizer_model()

    def load_tokenizer_model(self):
        data_path = self._file_service.get_data_path()
        tokenizer_path = os.path.join(data_path, 'tokenizer.model')
        if not os.path.exists(tokenizer_path):
            return

        self._tokenizer.Load(tokenizer_path)
        self._tokenizer_loaded = True

    def is_tokenizer_loaded(self) -> bool:
        return self._tokenizer_loaded

    @property
    def tokenizer(self):
        return self._tokenizer
