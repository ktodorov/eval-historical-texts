import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from overrides import overrides
from models.model_base import ModelBase


class ConditionalRandomField(ModelBase):
    def __init__(
            self,
            num_of_tags: int,
            device,
            context_emb: int,
            start_token_id: int,
            stop_token_id: int,
            pad_token_id: int,
            none_id: int,
            use_weighted_loss: bool):
        super().__init__()

        self._number_of_tags = num_of_tags
        self._device = device
        self.use_char = False
        self.context_emb = context_emb

        self.start_idx = start_token_id
        self.end_idx = stop_token_id
        self.pad_idx = pad_token_id
        self.none_idx = none_id

        self.use_weighted_loss = use_weighted_loss

        # initialize the following transition (anything never -> start. end never -> anything. Same thing for the padding label)
        init_transition = torch.randn(
            self._number_of_tags, self._number_of_tags, device=self._device)
        init_transition[:, self.start_idx] = -10000.0
        init_transition[self.end_idx, :] = -10000.0
        init_transition[:, self.pad_idx] = -10000.0
        init_transition[self.pad_idx, :] = -10000.0

        self._transition_matrix = nn.Parameter(init_transition)

        if self.use_weighted_loss:
            self._weighted_loss_matrix = torch.ones(
                self._number_of_tags, self._number_of_tags, device=self._device, requires_grad=False)
            self._weighted_loss_matrix[self.none_idx, :] = 1.1
            self._weighted_loss_matrix[:, self.none_idx] = 1.3
            self._weighted_loss_matrix[self.none_idx, self.none_idx] = 1

    @overrides
    def forward(
            self,
            rnn_features,
            lengths,
            targets,
            mask):
        """
        Calculate the negative log-likelihood
        :param rnn_features:
        :param lengths:
        :param targets:
        :param mask:
        :return:
        """
        all_scores = self.calculate_all_scores(rnn_features=rnn_features)
        batch_size = rnn_features.shape[0]
        if self.training and self.use_weighted_loss:
            weighted_scores = torch.ones(all_scores.shape, device=all_scores.device)
            for b in range(batch_size):
                for word_idx in range(lengths[b]):
                    for tag in range(self._number_of_tags):
                        weighted_scores[b, word_idx, :, tag] = all_scores[b, word_idx, :, tag] * self._weighted_loss_matrix[targets[b, word_idx], tag]
        else:
            weighted_scores = all_scores

        unlabed_score = self._forward_unlabeled(weighted_scores, lengths, targets)
        labeled_score = self._forward_labeled(
            weighted_scores, lengths, targets, mask)
        loss = unlabed_score - labeled_score

        _, decoded_tags = self._viterbi_decode(all_scores, lengths)
        return loss, decoded_tags

    def _forward_unlabeled(
            self,
            all_scores: torch.Tensor,
            word_seq_lens: torch.Tensor,
            targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the scores with the forward algorithm. Basically calculating the normalization term
        :param all_scores: (batch_size x max_seq_len x num_labels x num_labels) from (lstm scores + transition scores).
        :param word_seq_lens: (batch_size)
        :return: The score for all the possible structures.
        """
        batch_size, max_sequence_length, _, _ = all_scores.shape
        alpha = torch.zeros(
            (batch_size, max_sequence_length, self._number_of_tags),
            device=self._device)

        # the first position of all labels = (the transition from start - > all labels) + current emission
        alpha[:, 0, :] = all_scores[:, 0,  self.start_idx, :]

        for word_idx in range(1, max_sequence_length):
            previous_alpha = alpha[:, word_idx-1, :].unsqueeze(-1).expand(
                batch_size,
                self._number_of_tags,
                self._number_of_tags)

            before_log_sum_exp = previous_alpha + all_scores[:, word_idx, :, :]
            alpha[:, word_idx, :] = self._log_sum_exp(before_log_sum_exp, targets[:, word_idx])

        # batch_size x number_of_tags
        last_alpha = alpha.gather(1, word_seq_lens.view(batch_size, 1, 1).expand(
            batch_size, 1, self._number_of_tags)-1).view(batch_size, self._number_of_tags)
        last_alpha += self._transition_matrix[:, self.end_idx].view(
            1, self._number_of_tags).expand(batch_size, self._number_of_tags)
        last_alpha = self._log_sum_exp(last_alpha.view(
            batch_size, self._number_of_tags, 1), targets[:, -1]).view(batch_size)

        return torch.mean(last_alpha)

    def _forward_labeled(
            self,
            all_scores: torch.Tensor,
            word_seq_lens: torch.Tensor,
            targets: torch.Tensor,
            masks: torch.Tensor) -> torch.Tensor:
        '''
        Calculate the scores for the gold instances.
        :param all_scores: (batch, max_length, number_of_tags, number_of_tags)
        :param word_seq_lens: (batch, max_length)
        :param targets: (batch, max_length)
        :param masks: batch, max_length
        :return: sum of score for the gold sequences Shape: (batch_size)
        '''
        batchSize, max_sequence_length, _, _ = all_scores.shape

        # all the scores to current labels: batch, max_length, all_from_label?
        expanded_targets = targets.view(
            batchSize,
            max_sequence_length,
            1,
            1).expand(
                batchSize,
                max_sequence_length,
                self._number_of_tags,
                1)

        current_tag_scores = all_scores.gather(3, expanded_targets).squeeze(-1)

        if max_sequence_length != 1:
            middle_transition_scores = current_tag_scores[:, 1:, :].gather(
                2, targets[:, :(max_sequence_length - 1)].unsqueeze(-1)).squeeze(-1)

        start_transition_scores = current_tag_scores[:, 0, self.start_idx]
        endTagIds = targets.gather(1, word_seq_lens.unsqueeze(-1) - 1)

        expanded_transition_matrix = self._transition_matrix[:, self.end_idx].unsqueeze(0).expand(
            batchSize,
            self._number_of_tags)

        end_transition_scores = expanded_transition_matrix.gather(
            1, endTagIds).view(batchSize)

        score = start_transition_scores + end_transition_scores
        if max_sequence_length != 1:
            score += torch.sum(middle_transition_scores *
                               masks[:, 1:].long(), dim=1)

        return torch.mean(score)

    def calculate_all_scores(
            self,
            rnn_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate all scores by adding up the transition scores and emissions (from lstm).
        Basically, compute the scores for each edges between labels at adjacent positions.
        This score is later be used for forward-backward inference
        :param rnn_features: emission scores.
        :return:
        """
        batch_size, max_length, _ = rnn_features.shape

        expanded_transition_matrix = self._transition_matrix.view(
            1,
            1,
            self._number_of_tags,
            self._number_of_tags).expand(
                batch_size,
                max_length,
                self._number_of_tags,
                self._number_of_tags)

        expanded_rnn_features = rnn_features.unsqueeze(2).expand(
            batch_size,
            max_length,
            self._number_of_tags,
            self._number_of_tags)

        all_scores = (expanded_transition_matrix + expanded_rnn_features)
        return all_scores

    def decode(
            self,
            features,
            wordSeqLengths) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        all_scores = self.calculate_all_scores(features)
        _, decodeIdx = self._viterbi_decode(
            all_scores, wordSeqLengths)

        return decodeIdx

    def _viterbi_decode(
            self,
            all_scores: torch.Tensor,
            word_seq_lens: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use Viterbi to decode the instances given the scores and transition parameters
        :param all_scores: (batch_size x max_length x number_of_tags x number_of_tags)
        :param word_seq_lens: (batch_size)
        :return: the best scores as well as the predicted label ids.
               (batch_size) and (batch_size x max_length)
        """
        batch_size, max_length, _, _ = all_scores.shape

        score_records = torch.zeros(
            (batch_size, max_length, self._number_of_tags),
            device=self._device)

        indices_records = torch.zeros(
            (batch_size, max_length, self._number_of_tags),
            dtype=torch.int64,
            device=self._device)

        start_indices = torch.full(
            (batch_size, self._number_of_tags),
            self.start_idx,
            dtype=torch.int64,
            device=self._device)

        decoded_tags = torch.full(
            (batch_size, max_length),
            self.pad_idx,
            dtype=torch.long,
            device=self._device)

        # represent the best current score from the start, is the best
        score_records[:, 0, :] = all_scores[:, 0, self.start_idx, :]
        indices_records[:,  0, :] = start_indices
        for wordIdx in range(1, max_length):
            current_word_scores = score_records[:, wordIdx - 1, :].unsqueeze(-1).expand(
                batch_size, self._number_of_tags, self._number_of_tags)
            current_word_scores = current_word_scores + \
                all_scores[:, wordIdx, :, :]

            # the best previous label idx to current labels
            current_index_record = torch.argmax(
                current_word_scores, dim=1, keepdim=True)
            indices_records[:, wordIdx, :] = current_index_record.squeeze(1)
            score_records[:, wordIdx, :] = current_word_scores.gather(
                1, current_index_record).squeeze(1)

        word_seq_lens_indices = word_seq_lens.view(
            batch_size, 1, 1).expand(batch_size, 1, self._number_of_tags)
        lastScores = score_records.gather(
            1, word_seq_lens_indices - 1).squeeze(1)
        lastScores += self._transition_matrix[:, self.end_idx].unsqueeze(
            0).expand(batch_size, self._number_of_tags)

        last_indices = torch.argmax(lastScores, 1)
        bestScores = lastScores.gather(1, last_indices.unsqueeze(-1))

        for b, word_seq_len in enumerate(word_seq_lens):
            decoded_tags[b, word_seq_len-1] = last_indices[b]

            for distance2Last in range(1, word_seq_len):
                lastNIdxRecord = indices_records[b,
                                                 word_seq_len - distance2Last]
                decoded_tags[b, word_seq_len - distance2Last -
                             1] = lastNIdxRecord[decoded_tags[b, word_seq_len - distance2Last]]

        return bestScores, decoded_tags

    def _log_sum_exp(
            self,
            vec: torch.Tensor,
            current_targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log_sum_exp trick for the tensor.
        :param vec: [batchSize * from_label * to_label].
        :return: [batchSize * to_label]
        """
        max_scores, idx = torch.max(vec, 1)
        max_scores[max_scores == -float("Inf")] = 0
        expanded_max_scores = max_scores.unsqueeze(1).expand_as(vec)
        diff = (vec - expanded_max_scores).exp().sum(dim=1).log()
        result = max_scores + diff
        return result
