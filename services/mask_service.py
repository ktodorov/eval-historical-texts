import torch

from typing import Tuple

from services.tokenizer_service import TokenizerService
from services.arguments_service_base import ArgumentsServiceBase


class MaskService:
    def __init__(
            self,
            tokenizer_service: TokenizerService,
            arguments_service: ArgumentsServiceBase):

        self._tokenizer_service = tokenizer_service
        self._arguments_service = arguments_service

    def mask_tokens(
            self,
            inputs: torch.Tensor,
            lengths: torch.Tensor,
            mlm_probability=0.15) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

        :param inputs: inputs that will be masked and then forwarded to the model
        :type inputs: torch.Tensor
        :param mlm_probability: the probability for masking a token, defaults to 0.15
        :type mlm_probability: float, optional
        :return: returns the masked inputs as well as the labels for the masks
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        # TODO add check
        tokenizer = self._tokenizer_service.get_sub_tokenizer()

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = [tokenizer.get_special_tokens_mask(
            val, already_has_special_tokens=True) for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(
            special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # Padded tokens must not be used to compute loss
        for length in lengths:
            labels[length:] = -1

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
            tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long,
                                     device=self._arguments_service.get_argument('device'))
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
