import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from typing import List, Dict

from entities.metric import Metric
from enums.metric_type import MetricType
from enums.ner_type import NERType

from models.model_base import ModelBase
from models.ner_rnn.rnn_attention import RNNAttention

from services.arguments.ner_arguments_service import NERArgumentsService
from services.data_service import DataService
from services.metrics_service import MetricsService
from services.tokenizer_service import TokenizerService
from services.process.ner_process_service import NERProcessService


class NERRNNModel(ModelBase):
    def __init__(
            self,
            arguments_service: NERArgumentsService,
            data_service: DataService,
            metrics_service: MetricsService,
            process_service: NERProcessService,
            tokenizer_service: TokenizerService):
        super().__init__(data_service, arguments_service)

        self._arguments_service = arguments_service
        self.process_service = process_service

        self._include_pretrained = arguments_service.include_pretrained_model
        additional_size = arguments_service.pretrained_model_size if self._include_pretrained else 0
        self._learn_embeddings = arguments_service.learn_new_embeddings

        self._metrics_service = metrics_service
        self._metric_types = arguments_service.metric_types
        number_of_tags = process_service.get_labels_amount()

        # maps each token to an embedding_dim vector
        lstm_input_size = additional_size
        if self._learn_embeddings:
            self.embedding = nn.Embedding(
                tokenizer_service.vocabulary_size, arguments_service.embeddings_size)
            self.dropout = nn.Dropout(arguments_service.dropout)
            lstm_input_size += arguments_service.embeddings_size

        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(
            lstm_input_size, arguments_service.hidden_dimension, batch_first=True, bidirectional=True)

        if self._arguments_service.use_attention:
            attention_dimension = arguments_service.hidden_dimension * 2
            self.attention = RNNAttention(
                attention_dimension, attention_dimension, attention_dimension)

        # fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(
            arguments_service.hidden_dimension * 2, number_of_tags)


        self.hidden_dim = arguments_service.hidden_dimension
        self.tagset_size = number_of_tags

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.process_service.START_TAG, :] = -10000
        self.transitions.data[:, self.process_service.STOP_TAG] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def forward(self, input_batch, debug=False, **kwargs):
        sequences, targets, lengths, pretrained_representations = input_batch

        # apply the embedding layer that maps each token to its embedding
        # dim: batch_size x batch_max_len x embedding_dim
        if self._learn_embeddings:
            embedded = self.dropout(self.embedding(sequences))

            if self._include_pretrained:
                embedded = torch.cat(
                    (embedded, pretrained_representations), dim=2)
        else:
            embedded = pretrained_representations

        x_packed = pack_padded_sequence(embedded, lengths, batch_first=True)

        # run the LSTM along the sentences of length batch_max_len
        # dim: batch_size x batch_max_len x lstm_hidden_dim
        packed_output, hidden = self.lstm(x_packed)
        if isinstance(hidden, tuple):  # LSTM
            hidden = hidden[1]  # take the cell state

        # TODO: add bidirectional as argument
        if True:  # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]

        rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        if self._arguments_service.use_attention:
            energy, linear_combination = self.attention.forward(
                hidden, rnn_output, rnn_output)
            linear_combination = linear_combination.expand_as(rnn_output)

            rnn_output = linear_combination * rnn_output

        rnn_output = rnn_output.reshape(-1, rnn_output.shape[2])

        # apply the fully connected layer and obtain the output for each token
        # dim: batch_size*batch_max_len x num_tags
        output = self.fc(rnn_output)

        # dim: batch_size*batch_max_len x num_tags
        # return F.log_softmax(output, dim=1), targets


        forward_score = self._forward_alg(output)
        gold_score = self._score_sentence(output, targets.reshape(-1))
        return forward_score - gold_score, output, targets

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(self._arguments_service.device)
        # START_TAG has all of the score.
        init_alphas[0][self.process_service.START_TAG] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = torch.zeros((1, self.tagset_size)).to(self._arguments_service.device)  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t[:, next_tag] = self.log_sum_exp(next_tag_var).view(1)
            forward_var = alphas_t
        terminal_var = forward_var + self.transitions[self.process_service.STOP_TAG]
        alpha = self.log_sum_exp(terminal_var)
        return alpha

    def argmax(self, vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return idx.item()

    # Compute log sum exp in a numerically stable way for the forward algorithm
    def log_sum_exp(self, vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + \
            torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(self._arguments_service.device)
        start_tag = torch.tensor([self.process_service.START_TAG], dtype=torch.long, device=self._arguments_service.device)
        tags = torch.cat([start_tag, tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.process_service.STOP_TAG, tags[-1]]
        return score

    def calculate_accuracies(self, batch, outputs, output_characters=False) -> Dict[MetricType, float]:
        _, output, targets = outputs
        predictions = output.max(dim=1)[1].cpu().detach().numpy()

        targets = targets.reshape(-1).cpu().detach().numpy()

        mask = np.array((targets != -1), dtype=bool)
        predicted_labels = predictions[mask]
        target_labels = targets[mask]

        metrics = {}

        if MetricType.F1Score in self._metric_types:
            f1_score = self._metrics_service.calculate_f1_score(
                predicted_labels, target_labels)
            metrics[MetricType.F1Score] = f1_score

        return metrics, None

    def compare_metric(self, best_metric: Metric, new_metric: Metric) -> bool:
        if best_metric.is_new:
            return True

        return best_metric.get_accuracy_metric(MetricType.F1Score) <= new_metric.get_accuracy_metric(MetricType.F1Score)
