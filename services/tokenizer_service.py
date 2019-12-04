from transformers import PreTrainedTokenizer, BertTokenizer, XLNetTokenizer

from enums.configuration import Configuration
from services.arguments_service_base import ArgumentsServiceBase

class TokenizerService:
    def __init__(self, arguments_service: ArgumentsServiceBase):
        pretrained_weights = arguments_service.get_argument('pretrained_weights')
        configuration : Configuration = arguments_service.get_argument('configuration')

        if configuration == Configuration.KBert:
            self._tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        else:
            self._tokenizer = XLNetTokenizer.from_pretrained(pretrained_weights)

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer
