import torch
import torch.nn as nn

import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Dict, List
from overrides import overrides

from entities.metric import Metric
from entities.data_output_log import DataOutputLog
from entities.batch_representation import BatchRepresentation
from entities.options.rnn_encoder_options import RNNEncoderOptions
from entities.options.pretrained_representations_options import PretrainedRepresentationsOptions

from enums.metric_type import MetricType
from enums.tag_measure_averaging import TagMeasureAveraging
from enums.tag_measure_type import TagMeasureType
from enums.entity_tag_type import EntityTagType

from models.ner_rnn.rnn_encoder import RNNEncoder
from models.ner_rnn.conditional_random_field import ConditionalRandomField
from models.model_base import ModelBase

from services.arguments.ner_arguments_service import NERArgumentsService
from services.data_service import DataService
from services.metrics_service import MetricsService
from services.file_service import FileService
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.process.ner_process_service import NERProcessService


class NERPredictor(ModelBase):

    def __init__(
            self,
            arguments_service: NERArgumentsService,
            data_service: DataService,
            metrics_service: MetricsService,
            process_service: NERProcessService,
            tokenize_service: BaseTokenizeService,
            file_service: FileService):
        super().__init__(data_service, arguments_service)

        self._metrics_service = metrics_service
        self._process_service = process_service

        self.device = arguments_service.device
        self.number_of_tags = process_service.get_labels_amount()
        self._metric_types = arguments_service.metric_types

        self._entity_tag_types = arguments_service.entity_tag_types

        pretrained_options = PretrainedRepresentationsOptions(
            include_pretrained_model=arguments_service.include_pretrained_model,
            pretrained_model_size=arguments_service.pretrained_model_size,
            pretrained_weights=arguments_service.pretrained_weights,
            pretrained_max_length=arguments_service.pretrained_max_length,
            pretrained_model=arguments_service.pretrained_model,
            fine_tune_pretrained=arguments_service.fine_tune_pretrained,
            include_fasttext_model=arguments_service.include_fasttext_model,
            fasttext_model=arguments_service.fasttext_model,
            fasttext_model_size=arguments_service.fasttext_model_size)

        rnn_encoder_options = RNNEncoderOptions(
            device=self.device,
            pretrained_representations_options=pretrained_options,
            number_of_tags=self.number_of_tags,
            use_attention=arguments_service.use_attention,
            learn_new_embeddings=arguments_service.learn_new_embeddings,
            vocabulary_size=tokenize_service.vocabulary_size,
            embeddings_size=arguments_service.embeddings_size,
            dropout=arguments_service.dropout,
            hidden_dimension=arguments_service.hidden_dimension,
            bidirectional=arguments_service.bidirectional_rnn,
            number_of_layers=arguments_service.number_of_layers,
            merge_subword_embeddings=arguments_service.merge_subwords,
            learn_character_embeddings=arguments_service.learn_character_embeddings,
            character_embeddings_size=arguments_service.character_embeddings_size,
            character_hidden_size=arguments_service.character_hidden_size)

        self.rnn_encoder = RNNEncoder(
            file_service,
            rnn_encoder_options)

        self._pad_idx = self._process_service.pad_idx
        start_token_id = self._process_service.start_idx
        stop_token_id = self._process_service.stop_idx

        self._crf_layers = nn.ModuleList([
            ConditionalRandomField(
                num_of_tags=number_of_tags,
                device=arguments_service.device,
                context_emb=arguments_service.hidden_dimension,
                start_token_id=start_token_id,
                stop_token_id=stop_token_id,
                pad_token_id=self._pad_idx,
                none_id=self._process_service.get_entity_label('O', entity_tag_type),
                use_weighted_loss=arguments_service.use_weighted_loss)
            for i, (entity_tag_type, number_of_tags) in enumerate(self.number_of_tags.items())
        ])

        self.tag_measure_averages = [
            TagMeasureAveraging.Macro, TagMeasureAveraging.Micro, TagMeasureAveraging.Weighted]
        self.tag_measure_types = [TagMeasureType.Strict]

    @overrides
    def forward(self, batch_representation: BatchRepresentation):
        rnn_outputs, lengths = self.rnn_encoder.forward(
            batch_representation)

        losses: Dict[EntityTagType, torch.Tensor] = {}
        predictions: Dict[EntityTagType, torch.Tensor] = {}

        for i, entity_tag_type in enumerate(self._entity_tag_types):
            mask = self._create_mask(rnn_outputs[entity_tag_type], lengths)
            loss, prediction = self._crf_layers[i].forward(rnn_outputs[entity_tag_type], lengths, batch_representation.targets[entity_tag_type], mask)
            losses[entity_tag_type] = loss
            predictions[entity_tag_type] = prediction

        return predictions, losses, lengths

    @overrides
    def calculate_accuracies(
            self,
            batch: BatchRepresentation,
            outputs,
            output_characters=False) -> Dict[MetricType, float]:
        output, _, lengths = outputs
        targets = batch.targets

        metrics: Dict[str, float] = {}

        output_log = None
        if output_characters:
            output_log = DataOutputLog()

        for entity_tag_type in self._entity_tag_types:

            predictions = output[entity_tag_type].cpu().detach().numpy()
            current_targets = targets[entity_tag_type].cpu().detach().numpy()
            none_idx = self._process_service.get_entity_label('O', entity_tag_type)

            mask = np.where(((current_targets != self._pad_idx) & (current_targets != none_idx)), True, False)

            # if current batch has no targets other than O, we calculate the F1 score on those instead
            if mask.sum() == 0:
                mask = np.where((current_targets != self._pad_idx), True, False)

            predicted_labels = predictions[mask]
            target_labels = current_targets[mask]

            for tag_measure_averaging in self.tag_measure_averages:
                for tag_measure_type in self.tag_measure_types:
                    (precision_score, recall_score, f1_score, _) = self._metrics_service.calculate_precision_recall_fscore_support(
                        predicted_labels,
                        target_labels,
                        tag_measure_type,
                        tag_measure_averaging)

                    if MetricType.F1Score in self._metric_types:
                        key = self._create_measure_key(
                            MetricType.F1Score, tag_measure_averaging, tag_measure_type, entity_tag_type)
                        metrics[key] = f1_score

                    if MetricType.PrecisionScore in self._metric_types:
                        key = self._create_measure_key(
                            MetricType.PrecisionScore, tag_measure_averaging, tag_measure_type, entity_tag_type)
                        metrics[key] = precision_score

                    if MetricType.RecallScore in self._metric_types:
                        key = self._create_measure_key(
                            MetricType.RecallScore, tag_measure_averaging, tag_measure_type, entity_tag_type)
                        metrics[key] = recall_score

            if output_characters:
                for i in range(batch.batch_size):
                    predicted_string = ','.join(
                        [self._process_service.get_entity_by_label(predicted_label, entity_tag_type) for predicted_label in predictions[i][:lengths[i]]])
                    target_string = ','.join([self._process_service.get_entity_by_label(target_label, entity_tag_type)
                                            for target_label in current_targets[i][:lengths[i]]])

                    output_log.add_new_data(
                        output_data=predicted_string,
                        true_data=target_string)

        return metrics, output_log

    @overrides
    def compare_metric(self, best_metric: Metric, new_metric: Metric) -> bool:
        if best_metric.is_new:
            return True

        keys = [
            self._create_measure_key(
                MetricType.F1Score,
                TagMeasureAveraging.Macro,
                TagMeasureType.Strict,
                entity_tag_type)
            for entity_tag_type in self._entity_tag_types
        ]

        current_best_value = np.mean([
            best_metric.get_accuracy_metric(key) for key in keys
        ])

        new_value = np.mean([
            new_metric.get_accuracy_metric(key) for key in keys
        ])

        return current_best_value <= new_value

    def _create_measure_key(
            self,
            metric_type: MetricType,
            tag_measure_averaging: TagMeasureAveraging,
            tag_measure_type: TagMeasureType,
            entity_tag_type: EntityTagType):
        key = f'{metric_type.value}-{tag_measure_averaging.value}-{tag_measure_type.value}-{entity_tag_type.value}'
        return key

    def _create_mask(self, rnn_outputs: torch.Tensor, lengths: torch.Tensor):
        batch_size = rnn_outputs.size(0)
        sent_len = rnn_outputs.size(1)
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long).view(
            1, sent_len).expand(batch_size, sent_len).to(self.device)
        mask = torch.le(maskTemp, lengths.view(batch_size, 1).expand(
            batch_size, sent_len)).to(self.device)
        return mask
