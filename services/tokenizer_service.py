from transformers import PreTrainedTokenizer, BertTokenizer

from services.arguments_service_base import ArgumentsServiceBase

class TokenizerService:
    def __init__(self, arguments_service: ArgumentsServiceBase):
        pretrained_weights = arguments_service.get_argument('pretrained_weights')

        self._tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer
