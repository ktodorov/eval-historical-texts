from transformers import PreTrainedTokenizer, BertTokenizer, XLNetTokenizer

import sentencepiece as spm

from enums.configuration import Configuration
from services.arguments_service_base import ArgumentsServiceBase

class TokenizerService:
    def __init__(self, arguments_service: ArgumentsServiceBase):
        pretrained_weights = arguments_service.get_argument('pretrained_weights')
        configuration : Configuration = arguments_service.get_argument('configuration')

        if configuration == Configuration.KBert:
            self._tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        elif configuration == Configuration.XLNet:
            self._tokenizer = XLNetTokenizer.from_pretrained(pretrained_weights)
        elif configuration == Configuration.MultiFit:
            self._tokenizer = spm.SentencePieceProcessor()
            self._tokenizer.Load("data\\sentence-piece-models\\test_model.model")

    @property
    def tokenizer(self):
        return self._tokenizer
