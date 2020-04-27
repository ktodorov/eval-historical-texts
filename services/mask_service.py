import torch

from typing import Tuple

from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.arguments.pretrained_arguments_service import PretrainedArgumentsService


class MaskService:
    def __init__(
            self,
            tokenize_service: BaseTokenizeService,
            arguments_service: PretrainedArgumentsService):

        self._tokenize_service = tokenize_service
        self._arguments_service = arguments_service
        token_results = self._tokenize_service.encode_tokens(['[MASK]'])
        if token_results and len(token_results) > 0:
            self._mask_token_id = token_results[0]
        else:
            raise Exception('Mask token not found')

        self._masking_percentage = 0.8

    def mask_tokens(
            self,
            inputs: torch.Tensor,
            masks: torch.Tensor,
            lengths: torch.Tensor,
            mlm_probability=0.15) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

        :param inputs: inputs that will be masked and then forwarded to the model
        :type inputs: torch.Tensor
        :param masks: masks that correspond to the special tokens
        :type masks: torch.Tensor
        :param mlm_probability: the probability for masking a token, defaults to 0.15
        :type mlm_probability: float, optional
        :return: returns the masked inputs as well as the labels for the masks
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(
            labels.shape,
            mlm_probability,
            device=self._arguments_service.device)

        probability_matrix.masked_fill_(masks, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1 # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(
            torch.full(
                labels.shape,
                self._masking_percentage,
                device=self._arguments_service.device)
        ).bool() & masked_indices
        inputs[indices_replaced] = self._mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(
            torch.full(
                labels.shape,
                0.5,
                device=self._arguments_service.device)
        ).bool() & masked_indices & ~indices_replaced

        random_words = torch.randint(
            self._tokenize_service.vocabulary_size,
            labels.shape,
            dtype=torch.long,
            device=self._arguments_service.device)

        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
