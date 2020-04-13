import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from overrides import overrides

START_TAG = -3
STOP_TAG = -2


class LinearCRF(nn.Module):

    def __init__(
            self,
            num_of_tags: int,
            device):
        super().__init__()

        self.device = device
        # Matrix of transition parameters.  Entry i,j is the score of transitioning from i to j.
        self.tagset_size = num_of_tags
        # # We add 2 here, because of START_TAG and STOP_TAG
        # # transitions (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        init_transitions = torch.zeros(self.tagset_size+2, self.tagset_size+2)

        # impossible values for start_tag to be final and stop_tag to be first
        init_transitions[:, START_TAG] = -10000.0
        init_transitions[STOP_TAG, :] = -10000.0
        init_transitions[:, 0] = -10000.0
        init_transitions[0, :] = -10000.0
        # init_transitions = init_transitions

        self.transitions = nn.Parameter(init_transitions)

    def _calculate_PZ(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                masks: (batch, seq_len)
        """
        batch_size, seq_len, tag_size = feats.shape
        # print feats.view(seq_len, tag_size)
        assert(tag_size == self.tagset_size+2)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        # be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        # need to consider start
        scores = feats + \
            self.transitions.view(1, tag_size, tag_size).expand(
                ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # build iter
        seq_iter = enumerate(scores)
        # bat_size * from_target_size * to_target_size
        _, inivalues = next(seq_iter)
        # only need start from start_tag
        partition = inivalues[:, START_TAG, :].clone().view(
            batch_size, tag_size, 1)  # bat_size * to_target_size

        # add start score (from start to all tag, duplicate to batch_size)
        # partition = partition + self.transitions[START_TAG,:].view(1, tag_size, 1).expand(batch_size, tag_size, 1)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target

            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size,
                                                                  1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)
            # print cur_partition.data

            # (bat_size * from_target * to_target) -> (bat_size * to_target)
            # partition = utils.switch(partition, cur_partition, mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size)).view(bat_size, -1)
            mask_idx = mask[idx, :].view(
                batch_size, 1).expand(batch_size, tag_size)

            # effective updated partition part, only keep the partition value of mask value = 1
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            # let mask_idx broadcastable, to disable warning
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)

            # replace the partition where the maskvalue=1, other partition value keeps the same
            partition.masked_scatter_(mask_idx, masked_cur_partition)
        # until the last state, add transition score for all partition (and do log_sum_exp) then select the value in STOP_TAG
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(
            batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, STOP_TAG]
        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        batch_size, seq_len, tag_size = feats.shape
        assert(tag_size == self.tagset_size+2)
        # calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        # mask to (seq_len, batch_size)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        # be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        # need to consider start
        scores = feats + \
            self.transitions.view(1, tag_size, tag_size).expand(
                ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        sequences_iterator = enumerate(scores)
        # record the position of best score
        back_points = list()
        partition_history = list()
        # reverse mask (bug for mask = 1- mask, use this as alternative choice)
        # mask = 1 + (-1)*mask
        mask = (1 - mask.long()).bool()

        # bat_size * from_target_size * to_target_size
        _, inivalues = next(sequences_iterator)

        # only need start from start_tag
        partition = inivalues[:, START_TAG, :].clone().view(
            batch_size, tag_size)  # bat_size * to_target_size
        partition_history.append(partition)

        # iter over last scores
        for idx, cur_values in sequences_iterator:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: batch_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size,
                                                                  1).expand(batch_size, tag_size, tag_size)
            # forscores, cur_bp = torch.max(cur_values[:,:-2,:], 1) # do not consider START_TAG/STOP_TAG
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)
            # cur_bp: (batch_size, tag_size) max source score position in current tag
            # set padded label as 0, which will be filtered in post processing
            cur_bp.masked_fill_(mask[idx].view(
                batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)

        # add score to final STOP_TAG
        partition_history = torch.cat(partition_history, 0).view(
            seq_len, batch_size, -1).transpose(1, 0).contiguous()  # (batch_size, seq_len. tag_size)
        # get the last position for each sentences, and select the last partitions using gather()
        last_position = length_mask.view(
            batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(
            partition_history, 1, last_position).view(batch_size, tag_size, 1)
        # calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(
            1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long()
        pad_zero = pad_zero.to(self.device)
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(
            seq_len, batch_size, tag_size)

        # select end ids in STOP_TAG
        pointer = last_bp[:, STOP_TAG]
        insert_last = pointer.contiguous().view(
            batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()

        # move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        back_points.scatter_(1, last_position, insert_last)
        back_points = back_points.transpose(1, 0).contiguous()

        # decode from the end, padded position ids are 0, which will be filtered if following evaluation
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        decode_idx = decode_idx.to(self.device)
        decode_idx[-1] = pointer.detach()
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(
                back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.detach().view(batch_size)

        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def forward(self, feats, mask):
        path_score, best_path = self._viterbi_decode(feats, mask)
        return path_score, best_path

    def _score_sentence(self, scores, mask, targets):
        """
            input:
                scores: variable (seq_len, batch, tag_size, tag_size)
                mask: (batch, seq_len)
                tags: tensor  (batch, seq_len)
            output:
                score: sum of score for gold sequences within whole batch
        """
        # Gives the score of a provided tag sequence
        seq_len, batch_size, tag_size, _ = scores.shape
        # convert tag value into a new format, recorded label bigram information to index
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        new_tags = new_tags.to(self.device)
        for idx in range(seq_len):
            if idx == 0:
                # start -> first score
                new_tags[:, 0] = (tag_size - 2)*tag_size + targets[:, 0]

            else:
                new_tags[:, idx] = targets[:, idx-1]*tag_size + targets[:, idx]

        # transition for label to STOP_TAG
        end_transition = self.transitions[:, STOP_TAG].contiguous().view(
            1, tag_size).expand(batch_size, tag_size)

        # length for batch,  last word position = length - 1
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        # index the label id of last word
        end_ids = torch.gather(targets, 1, length_mask - 1)

        # index the transition score for end_id to STOP_TAG
        end_energy = torch.gather(end_transition, 1, end_ids)

        # convert tag as (seq_len, batch_size, 1)
        new_tags = new_tags.transpose(
            1, 0).contiguous().view(seq_len, batch_size, 1)
        # need convert tags id to search from 400 positions of scores
        tg_energy = torch.gather(scores.view(
            seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)  # seq_len * bat_size
        # mask transpose to (seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        # ## calculate the score from START_TAG to first label
        # start_transition = self.transitions[START_TAG,:].view(1, tag_size).expand(batch_size, tag_size)
        # start_energy = torch.gather(start_transition, 1, tags[0,:])

        # add all score together
        # gold_score = start_energy.sum() + tg_energy.sum() + end_energy.sum()
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, targets):
        # nonegative log likelihood
        forward_score, scores = self._calculate_PZ(feats, mask)
        gold_score = self._score_sentence(scores, mask, targets)
        return forward_score - gold_score

# Compute log sum exp in a numerically stable way for the forward algorithm


def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(
        vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)
