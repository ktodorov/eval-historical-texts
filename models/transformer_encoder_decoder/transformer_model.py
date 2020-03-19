from typing import Dict

import numpy as np

import torch
from torch import nn

from enums.metric_type import MetricType

from entities.metric import Metric

from models.model_base import ModelBase
from models.transformer_encoder_decoder.transformer_encoder import TransformerEncoder
from models.transformer_encoder_decoder.transformer_decoder import TransformerDecoder


from services.arguments.transformer_arguments_service import TransformerArgumentsService
from services.data_service import DataService
from services.metrics_service import MetricsService
from services.vocabulary_service import VocabularyService
from services.log_service import LogService
from services.tokenizer_service import TokenizerService


class TransformerModel(ModelBase):
    def __init__(
            self,
            arguments_service: TransformerArgumentsService,
            data_service: DataService,
            vocabulary_service: VocabularyService,
            metrics_service: MetricsService,
            log_service: LogService,
            tokenizer_service: TokenizerService):
        super().__init__(data_service)

        self._vocabulary_service = vocabulary_service
        self._metrics_service = metrics_service
        self._log_service = log_service
        self._tokenizer_service = tokenizer_service

        self._device = arguments_service.device
        self._metric_types = arguments_service.metric_types

        self.encoder = TransformerEncoder(
            input_dim=arguments_service.pretrained_vocabulary_size,
            hid_dim=arguments_service.hidden_dimension,
            n_layers=arguments_service.number_of_layers,
            n_heads=arguments_service.number_of_heads,
            pf_dim=arguments_service.hidden_dimension,
            dropout=arguments_service.dropout,
            device=arguments_service.device,
            include_pretrained=arguments_service.include_pretrained_model,
            pretrained_hidden_size=arguments_service.pretrained_model_size,
            max_length=5000)

        self.decoder = TransformerDecoder(
            output_dim=vocabulary_service.vocabulary_size(),
            hid_dim=arguments_service.hidden_dimension,
            n_layers=arguments_service.number_of_layers,
            n_heads=arguments_service.number_of_heads,
            pf_dim=arguments_service.hidden_dimension,
            dropout=arguments_service.dropout,
            device=arguments_service.device,
            include_pretrained=arguments_service.include_pretrained_model,
            pretrained_hidden_size=arguments_service.pretrained_model_size,
            max_length=5000)

        self.src_pad_idx = 0
        self.trg_pad_idx = vocabulary_service.pad_token
        self.device = arguments_service.device

        self.initialize_weights()

    def initialize_weights(self):
        if hasattr(self, 'weight') and self.weight.dim() > 1:
            nn.init.xavier_uniform_(self.weight.data)

    def make_src_mask(self, src):

        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):

        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)

        # trg_pad_mask = [batch size, 1, trg len, 1]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones(
            (trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, input_batch, debug=False, **kwargs):
        source, targets, lengths, pretrained_representations, _ = input_batch

        # source = source[:, :-1]
        cut_targets = targets[:, :-1]

        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(source)
        trg_mask = self.make_trg_mask(cut_targets)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(source, src_mask, pretrained_representations)

        # enc_src = [batch size, src len, hid dim]

        output, _ = self.decoder(cut_targets, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, targets, lengths

    def calculate_accuracies(self, batch, outputs, output_characters=False) -> Dict[MetricType, float]:
        output, targets, _ = outputs
        output_dim = output.shape[-1]
        predictions = output.max(dim=2)[1].cpu().detach().numpy()

        targets = targets[:,1:].cpu().detach().numpy()
        predicted_characters = []
        target_characters = []

        for i in range(targets.shape[0]):
            indices = np.array(
                (targets[i] != self._vocabulary_service.pad_token), dtype=bool)
            predicted_characters.append(predictions[i][indices])
            target_characters.append(targets[i][indices])

        metrics = {}

        if MetricType.JaccardSimilarity in self._metric_types:
            predicted_tokens = [self._vocabulary_service.ids_to_string(
                x) for x in predicted_characters]
            target_tokens = [self._vocabulary_service.ids_to_string(
                x) for x in target_characters]
            jaccard_score = np.mean([self._metrics_service.calculate_jaccard_similarity(
                target_tokens[i], predicted_tokens[i]) for i in range(len(predicted_tokens))])

            metrics[MetricType.JaccardSimilarity] = jaccard_score

        character_results = None
        if MetricType.LevenshteinDistance in self._metric_types:
            predicted_strings = [self._vocabulary_service.ids_to_string(
                x) for x in predicted_characters]
            target_strings = [self._vocabulary_service.ids_to_string(
                x) for x in target_characters]

            levenshtein_distance = np.mean([self._metrics_service.calculate_normalized_levenshtein_distance(
                predicted_strings[i], target_strings[i]) for i in range(len(predicted_strings))])

            metrics[MetricType.LevenshteinDistance] = levenshtein_distance

            if output_characters:
                character_results = []
                _, _, lengths, _, ocr_texts_tensor = batch
                ocr_texts = ocr_texts_tensor.cpu().detach().tolist()
                for i in range(len(ocr_texts)):
                    input_string = self._vocabulary_service.ids_to_string(
                        ocr_texts[i])

                    character_results.append(
                        [input_string, predicted_strings[i], target_strings[i]])

        return metrics, character_results

    def compare_metric(self, best_metric: Metric, new_metric: Metric) -> bool:
        if best_metric.is_new:
            return True

        best_loss = round(best_metric.get_current_loss(), 2)
        new_loss = round(new_metric.get_current_loss(), 2)

        if best_loss == new_loss:
            best_levenshtein = best_metric.get_accuracy_metric(MetricType.LevenshteinDistance)
            new_levenshtein = new_metric.get_accuracy_metric(MetricType.LevenshteinDistance)
            return new_levenshtein <= best_levenshtein
        else:
            return best_loss > new_loss
