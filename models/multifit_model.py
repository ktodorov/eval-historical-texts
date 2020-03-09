import torch
from torch import nn
import random
import numpy as np

from typing import List, Dict

from entities.model_checkpoint import ModelCheckpoint
from entities.metric import Metric

from enums.metric_type import MetricType

from models.model_base import ModelBase

from services.arguments.postocr_arguments_service import PostOCRArgumentsService
from services.data_service import DataService
from services.tokenizer_service import TokenizerService
from services.metrics_service import MetricsService
from services.log_service import LogService

from models.sequence_encoder import SequenceEncoder
from models.sequence_decoder import SequenceDecoder


class MultiFitModel(ModelBase):
    def __init__(
            self,
            arguments_service: PostOCRArgumentsService,
            data_service: DataService,
            tokenizer_service: TokenizerService,
            metrics_service: MetricsService,
            log_service: LogService):
        super(MultiFitModel, self).__init__(data_service)

        self._metrics_service = metrics_service
        self._log_service = log_service
        self._tokenizer_service = tokenizer_service

        self._device = arguments_service.device
        self._metric_types = arguments_service.metric_types

        self._output_dimension = arguments_service.pretrained_vocabulary_size

        self._teacher_forcing_ratio = arguments_service.teacher_forcing_ratio

        self._encoder = SequenceEncoder(
            embedding_size=arguments_service.encoder_embedding_size,
            input_size=arguments_service.pretrained_vocabulary_size,
            hidden_dimension=arguments_service.hidden_dimension,
            number_of_layers=arguments_service.number_of_layers,
            dropout=arguments_service.dropout,
            include_pretrained=arguments_service.include_pretrained_model,
            pretrained_hidden_size=arguments_service.pretrained_model_size,
            learn_embeddings=arguments_service.learn_new_embeddings
        )

        self._decoder = SequenceDecoder(
            embedding_size=arguments_service.decoder_embedding_size,
            output_dimension=self._output_dimension,
            hidden_dimension=arguments_service.hidden_dimension,
            number_of_layers=arguments_service.number_of_layers,
            dropout=arguments_service.dropout
        )

        self.apply(self.init_weights)
        self._ignore_index = 0

    @staticmethod
    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, input_batch, debug=False, **kwargs):
        source, targets, lengths, pretrained_representations = input_batch

        (batch_size, trg_len) = targets.shape
        trg_vocab_size = self._output_dimension

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len,
                              trg_vocab_size, device=self._device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        context = self._encoder.forward(
            source, lengths, pretrained_representations, debug=debug)

        hidden = context
        context = context.permute(1, 0, 2)

        # first input to the decoder is the <sos> tokens
        input = targets[:, 0]

        for t in range(0, trg_len):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden = self._decoder.forward(input, hidden, context)

            outputs[:, t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < self._teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            if teacher_force:
                input = targets[:, t]
            else:
                # get the highest predicted token from our predictions
                top1 = output.argmax(1)
                input = top1

        return outputs, targets, lengths

    def calculate_accuracies(self, batch, outputs, output_characters=False) -> Dict[MetricType, float]:
        output, targets, _ = outputs
        output_dim = output.shape[-1]
        predictions = output.max(dim=2)[1].cpu().detach().numpy()

        targets = targets.cpu().detach().numpy()
        predicted_characters = []
        target_characters = []

        for i in range(targets.shape[0]):
            indices = np.array((targets[i] != 0), dtype=bool)
            predicted_characters.append(predictions[i][indices])
            target_characters.append(targets[i][indices])

        metrics = {}

        if MetricType.JaccardSimilarity in self._metric_types:
            predicted_tokens = [self._tokenizer_service.decode_tokens(
                x) for x in predicted_characters]
            target_tokens = [self._tokenizer_service.decode_tokens(
                x) for x in target_characters]
            jaccard_score = np.mean([self._metrics_service.calculate_jaccard_similarity(
                target_tokens[i], predicted_tokens[i]) for i in range(len(predicted_tokens))])

            metrics[MetricType.JaccardSimilarity] = jaccard_score

        character_results = None
        if MetricType.LevenshteinDistance in self._metric_types:
            predicted_strings = [self._tokenizer_service.decode_string(
                x) for x in predicted_characters]
            target_strings = [self._tokenizer_service.decode_string(
                x) for x in target_characters]

            levenshtein_distance = np.mean([self._metrics_service.calculate_normalized_levenshtein_distance(
                predicted_strings[i], target_strings[i]) for i in range(len(predicted_strings))])

            metrics[MetricType.LevenshteinDistance] = levenshtein_distance

            if output_characters:
                character_results = []
                source, _, lengths, _ = batch
                for i in range(source.shape[0]):
                    source_character_ids = source[i][:lengths[i]].cpu(
                    ).detach().tolist()
                    input_string = self._tokenizer_service.decode_string(
                        source_character_ids)

                    character_results.append(
                        [input_string, predicted_strings[i], target_strings[i]])

        return metrics, character_results

    def compare_metric(self, best_metric: Metric, new_metric: Metric) -> bool:
        if best_metric.is_new:
            return True

        best_jaccard = round(best_metric.get_accuracy_metric(
            MetricType.JaccardSimilarity), 2)
        new_jaccard = round(new_metric.get_accuracy_metric(
            MetricType.JaccardSimilarity), 2)

        if best_jaccard == new_jaccard:
            best_levenshtein = best_metric.get_accuracy_metric(
                MetricType.LevenshteinDistance)
            new_levenshtein = new_metric.get_accuracy_metric(
                MetricType.LevenshteinDistance)
            new_is_better = new_levenshtein < best_levenshtein
        else:
            new_is_better = new_jaccard > best_jaccard

        return new_is_better
