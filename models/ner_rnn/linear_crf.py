import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from overrides import overrides

def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))

class LinearCRF(nn.Module):

    def __init__(
            self,
            num_of_tags: int,
            device,
            start_tag: int,
            stop_tag: int,
            pad_tag: int):
        super().__init__()

        self.device = device
        self.tagset_size = num_of_tags

        self.start_tag = start_tag
        self.stop_tag = stop_tag
        self.pad_tag = pad_tag

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.start_tag, :] = -10000
        self.transitions.data[:, self.stop_tag] = -10000
        self.transitions.data[:, self.pad_tag] = -10000
        self.transitions.data[self.pad_tag, :] = -10000
        self.transitions.data[self.pad_tag, self.stop_tag] = 0
        self.transitions.data[self.pad_tag, self.pad_tag] = 0

    def _forward_alg(self, features, mask):
        # initialize forward variables in log space
        batch_size = features.shape[0]
        score = torch.Tensor(batch_size, self.tagset_size).fill_(-10000).to(self.device) # [B, C]
        score[:, self.start_tag] = 0.
        transitions = self.transitions.unsqueeze(0) # [1, C, C]
        for t in range(features.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            emit_t = features[:, t].unsqueeze(2) # [B, C, 1]
            score_t = score.unsqueeze(1) + emit_t + transitions # [B, 1, C] -> [B, C, C]
            score_t = log_sum_exp(score_t) # [B, C, C] -> [B, C]
            score = score_t * mask_t + score * (1 - mask_t)
        score = log_sum_exp(score + self.transitions[self.stop_tag])
        return score # partition function

    def _score_sentence(self, features, targets, mask):
        # Gives the score of a provided tag sequence
        score = torch.zeros(features.shape[0], device=self.device)

        features = features.unsqueeze(3)
        transitions = self.transitions.unsqueeze(2)
        sequence_length = features.size(1) - 1
        for t in range(sequence_length): # recursion through the sequence
            # current_tag, next_tag = targets[t], targets[t+1]
            mask_t = mask[:, t]
            emit_t = torch.cat([features[t, targets[t + 1]] for features, targets in zip(features, targets)])
            trans_t = torch.cat([transitions[targets[t + 1], targets[t]] for targets in targets])
            score += (emit_t + trans_t) * mask_t

        targets_length = mask.sum(1).long().unsqueeze(1) - 1
        last_tag = targets.gather(1, targets_length).squeeze(1)
        score += self.transitions[self.stop_tag, last_tag]
        return score

    def _viterbi_decode(self, features, mask):
        # initialize backpointers and viterbi variables in log space
        bptr = torch.LongTensor().to(self.device)
        batch_size = features.shape[0]
        score = torch.Tensor(batch_size, self.tagset_size).fill_(-10000).to(self.device)
        score[:, self.start_tag] = 0.

        for t in range(features.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            score_t = score.unsqueeze(1) + self.transitions # [B, 1, C] -> [B, C, C]
            score_t, bptr_t = score_t.max(2) # best previous scores and targets
            score_t += features[:, t] # plus emission scores
            bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = score_t * mask_t + score * (1 - mask_t)
        score += self.transitions[self.stop_tag]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(batch_size):
            i = best_tag[b] # best tag
            j = int(mask[b].sum().item())
            for bptr_t in reversed(bptr[b][:j]):
                i = bptr_t[i]
                best_path[b].append(i)
            best_path[b].pop()
            best_path[b].reverse()

        return best_path

    def neg_log_likelihood(self, rnn_outputs, targets, mask):
        # features = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(rnn_outputs, mask)
        gold_score = self._score_sentence(rnn_outputs, targets, mask)
        return torch.mean(forward_score - gold_score)

    def forward(self, rnn_outputs, mask):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        # lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        tag_seq = self._viterbi_decode(rnn_outputs, mask)
        return tag_seq