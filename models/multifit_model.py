import torch
from torch import nn
import random
import numpy as np

from typing import List, Dict

from entities.model_checkpoint import ModelCheckpoint
from entities.metric import Metric

from enums.metric_type import MetricType

from models.model_base import ModelBase

from services.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService
from services.tokenizer_service import TokenizerService
from services.metrics_service import MetricsService

from models.multifit_encoder import MultiFitEncoder
from models.multifit_decoder import MultiFitDecoder


class MultiFitModel(ModelBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            data_service: DataService,
            tokenizer_service: TokenizerService,
            metrics_service: MetricsService):
        super(MultiFitModel, self).__init__(data_service, arguments_service)

        self._metrics_service = metrics_service

        self._device = arguments_service.get_argument('device')
        self._metric_types: List[MetricType] = arguments_service.get_argument(
            'metric_types')
        self._tokenizer_service = tokenizer_service

        self._output_dimension = self._arguments_service.get_argument(
            'sentence_piece_vocabulary_size')

        self._include_pretrained_model = False

        self._encoder = MultiFitEncoder(
            embedding_size=self._arguments_service.get_argument(
                'embedding_size'),
            input_size=self._arguments_service.get_argument(
                'sentence_piece_vocabulary_size'),
            hidden_dimension=self._arguments_service.get_argument(
                'hidden_dimension'),
            number_of_layers=self._arguments_service.get_argument(
                'number_of_layers'),
            dropout=self._arguments_service.get_argument('dropout'),
            include_bert=self._include_pretrained_model,
            pretrained_weights=arguments_service.get_argument(
                'pretrained_weights')
        )

        self._decoder = MultiFitDecoder(
            embedding_size=self._arguments_service.get_argument(
                'embedding_size'),
            output_dimension=self._output_dimension,
            hidden_dimension=self._arguments_service.get_argument(
                'hidden_dimension') * 2,
            number_of_layers=self._arguments_service.get_argument(
                'number_of_layers'),
            dropout=self._arguments_service.get_argument('dropout')
        )

        self.apply(self.init_weights)
        self._ignore_index = 0

    @staticmethod
    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, input_batch, **kwargs):
        source, targets, lengths = input_batch

        (batch_size, trg_len) = targets.shape
        trg_vocab_size = self._output_dimension

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len,
                              trg_vocab_size, device=self._device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self._encoder.forward(
            source, lengths)

        # first input to the decoder is the <sos> tokens
        input = targets[:, 0]
        teacher_forcing_ratio = 0.5

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self._decoder.forward(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            for i in range(batch_size):
                if targets[i, t] == self._ignore_index:
                    output[i] = self._ignore_index

            outputs[:, t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            if teacher_force:
                input = targets[:, t]
            else:
                # get the highest predicted token from our predictions
                top1 = output.argmax(1)
                input = top1

        return outputs, targets, lengths

    def calculate_accuracies(self, batch, outputs, print_characters=False) -> Dict[MetricType, float]:
        output, targets, _ = outputs
        output_dim = output.shape[-1]
        predicted_characters = output.reshape(-1, output_dim).max(dim=1)[
            1].cpu().detach().numpy()

        target_characters = targets.reshape(-1).cpu().detach().numpy()
        indices = np.array((target_characters != 0), dtype=bool)

        target_characters = target_characters[indices].tolist()
        predicted_characters = predicted_characters[indices].tolist()

        metrics = {}

        if MetricType.JaccardSimilarity in self._metric_types:
            predicted_tokens = self._tokenizer_service.decode_tokens(
                predicted_characters)
            target_tokens = self._tokenizer_service.decode_tokens(
                target_characters)
            jaccard_score = self._metrics_service.calculate_jaccard_similarity(target_tokens, predicted_tokens)

            # if print_characters:
            #     print(f'Predicted:\n{predicted_string.encode("utf-8")}')
            #     print('---------------------------------------------------------')
            #     print(f'Target:\n{target_string.encode("utf-8")}')
            #     print('---------------------------------------------------------')

            metrics[MetricType.JaccardSimilarity] = jaccard_score

        if MetricType.LevenshteinDistance in self._metric_types:
            predicted_string = self._tokenizer_service.decode_string(
                predicted_characters)
            target_string = self._tokenizer_service.decode_string(
                target_characters)

            levenshtein_distance = self._metrics_service.calculate_levenshtein_distance(predicted_string, target_string)

            metrics[MetricType.LevenshteinDistance] = levenshtein_distance

        return metrics

    def compare_metric(self, best_metric: Metric, new_metrics: Metric) -> bool:
        if best_metric.is_new or best_metric.get_current_loss() > new_metrics.get_current_loss():
            return True

        return False
