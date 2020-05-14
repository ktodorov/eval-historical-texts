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
from enums.tag_metric import TagMetric
from enums.entity_tag_type import EntityTagType
from enums.word_feature import WordFeature

from models.ner_rnn.rnn_encoder import RNNEncoder
from models.ner_rnn.conditional_random_field import ConditionalRandomField
from models.model_base import ModelBase

from services.arguments.ner_arguments_service import NERArgumentsService
from services.data_service import DataService
from services.metrics_service import MetricsService
from services.file_service import FileService
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.process.ner_process_service import NERProcessService
from services.tag_metrics_service import TagMetricsService


class NERPredictor(ModelBase):

    def __init__(
            self,
            arguments_service: NERArgumentsService,
            data_service: DataService,
            metrics_service: MetricsService,
            process_service: NERProcessService,
            tokenize_service: BaseTokenizeService,
            file_service: FileService,
            tag_metrics_service: TagMetricsService):
        super().__init__(data_service, arguments_service)

        self._metrics_service = metrics_service
        self._process_service = process_service
        self._arguments_service = arguments_service
        self._tag_metrics_service = tag_metrics_service

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
            fine_tune_after_convergence=arguments_service.fine_tune_after_convergence,
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
            character_hidden_size=arguments_service.character_hidden_size,
            learn_manual_features=arguments_service.use_manual_features,
            manual_features_count=len(WordFeature))

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
                none_id=self._process_service.get_entity_label(
                    'O', entity_tag_type),
                use_weighted_loss=arguments_service.use_weighted_loss)
            for i, (entity_tag_type, number_of_tags) in enumerate(self.number_of_tags.items())
        ])

        self.tag_measure_averages = [TagMeasureAveraging.Micro]
        self.tag_measure_types = [
            TagMeasureType.Partial, TagMeasureType.Strict]

        self.metric_log_key = self._create_measure_key(
            TagMetric.F1ScoreMicro,
            TagMeasureType.Partial,
            EntityTagType.LiteralCoarse if EntityTagType.LiteralCoarse in self._entity_tag_types else self._entity_tag_types[
                0],
            'all')

    @overrides
    def forward(self, batch_representation: BatchRepresentation):
        rnn_outputs, lengths = self.rnn_encoder.forward(
            batch_representation)

        losses: Dict[EntityTagType, torch.Tensor] = {}
        predictions: Dict[EntityTagType, torch.Tensor] = {}

        for i, entity_tag_type in enumerate(self._entity_tag_types):
            mask = self._create_mask(rnn_outputs[entity_tag_type], lengths)
            loss, prediction = self._crf_layers[i].forward(
                rnn_outputs[entity_tag_type], lengths, batch_representation.targets[entity_tag_type], mask)
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

            main_entities = self._process_service.get_main_entities(
                entity_tag_type)
            prediction_tags = []
            target_tags = []
            for b in range(batch.batch_size):
                current_prediction_tags = [self._process_service.get_entity_by_label(
                    token, entity_tag_type) for token in predictions[b][:lengths[b]] if token != self._pad_idx]
                prediction_tags.append(current_prediction_tags)

                current_target_tags = [self._process_service.get_entity_by_label(
                    token, entity_tag_type) for token in current_targets[b][:lengths[b]] if token != self._pad_idx]
                target_tags.append(current_target_tags)

            if self.training:
                results, results_per_type = self._tag_metrics_service.calculate_batch(
                    prediction_tags, target_tags, main_entities)
                self.update_metrics(results, results_per_type,
                                    metrics, entity_tag_type)
            else:
                self._tag_metrics_service.add_predictions(
                    prediction_tags, target_tags, main_entities, entity_tag_type)

            if output_characters:
                for b in range(batch.batch_size):
                    predicted_string = ','.join(prediction_tags[b])
                    target_string = ','.join(target_tags[b])

                    output_log.add_new_data(
                        output_data=predicted_string,
                        true_data=target_string)

        return metrics, output_log

    def update_metrics(self, results, results_per_type, metrics, entity_tag_type):
        training_metrics = [TagMetric.F1ScoreMicro,
                            TagMetric.PrecisionMicro,
                            TagMetric.RecallMicro]

        for result_type, type_results in results.items():
            for metric_type, results_per_metric in type_results.items():
                if not self.training or metric_type in training_metrics:
                    measure_key = self._create_measure_key(
                        metric_type, result_type, entity_tag_type, 'all')
                    metrics[measure_key] = results_per_metric

        for entity_type, results_per_entity in results_per_type.items():
            for result_type, type_results in results_per_entity.items():
                for metric_type, results_per_metric in type_results.items():
                    if not self.training or metric_type in training_metrics:
                        measure_key = self._create_measure_key(
                            metric_type, result_type, entity_tag_type, entity_type)
                        metrics[measure_key] = results_per_metric

        return metrics

    @overrides
    def compare_metric(self, best_metric: Metric, new_metric: Metric) -> bool:
        if best_metric.is_new:
            return True

        keys = [
            self._create_measure_key(
                TagMetric.F1ScoreMicro,
                TagMeasureType.Partial,
                entity_tag_type,
                'all')
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
            metric_type: TagMetric,
            tag_measure_type: TagMeasureType,
            entity_tag_type: EntityTagType,
            entity_str: str):
        key = f'{metric_type.value}-{tag_measure_type.value}-{entity_str}-{entity_tag_type.value}'
        return key

    def _create_mask(self, rnn_outputs: torch.Tensor, lengths: torch.Tensor):
        batch_size = rnn_outputs.size(0)
        sent_len = rnn_outputs.size(1)
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long).view(
            1, sent_len).expand(batch_size, sent_len).to(self.device)
        mask = torch.le(maskTemp, lengths.view(batch_size, 1).expand(
            batch_size, sent_len)).to(self.device)
        return mask

    @overrides
    def optimizer_parameters(self):
        if not self._arguments_service.fine_tune_learning_rate or not self.rnn_encoder._embedding_layer._include_pretrained:
            return self.parameters()

        pretrained_layer_parameters = self.rnn_encoder._embedding_layer._pretrained_layer.parameters()
        model_parameters = [
            param
            for param in self.parameters()
            if param not in pretrained_layer_parameters
        ]

        result = [
            {
                'params': model_parameters
            },
            {
                'params': pretrained_layer_parameters,
                'lr': self._arguments_service.fine_tune_learning_rate
            }
        ]

        return result

    @overrides
    def calculate_overall_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        overall_stats = self._tag_metrics_service.calculate_overall_stats()
        for entity_tag_type, (results, results_per_type) in overall_stats.items():
            self.update_metrics(results, results_per_type,
                                metrics, entity_tag_type)

        self._tag_metrics_service.reset()

        return metrics
