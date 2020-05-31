import torch
from torch import nn
import numpy as np

from typing import List, Dict
from overrides import overrides

from entities.model_checkpoint import ModelCheckpoint
from entities.metric import Metric
from entities.batch_representation import BatchRepresentation
from entities.options.embedding_layer_options import EmbeddingLayerOptions
from entities.options.pretrained_representations_options import PretrainedRepresentationsOptions
from entities.data_output_log import DataOutputLog

from enums.metric_type import MetricType
from enums.embedding_type import EmbeddingType

from models.model_base import ModelBase

from services.arguments.postocr_arguments_service import PostOCRArgumentsService
from services.data_service import DataService
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.metrics_service import MetricsService
from services.log_service import LogService
from services.vocabulary_service import VocabularyService
from services.file_service import FileService
from services.process.ocr_character_process_service import OCRCharacterProcessService

from models.rnn_encoder_decoder.sequence_encoder import SequenceEncoder
from models.rnn_encoder_decoder.sequence_decoder import SequenceDecoder
from models.rnn_encoder_decoder.sequence_generator import SequenceGenerator
from models.embedding.embedding_layer import EmbeddingLayer

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class SequenceModel(ModelBase):
    def __init__(
            self,
            arguments_service: PostOCRArgumentsService,
            data_service: DataService,
            metrics_service: MetricsService,
            vocabulary_service: VocabularyService,
            file_service: FileService,
            process_service: OCRCharacterProcessService,
            log_service: LogService):
        super(SequenceModel, self).__init__(data_service, arguments_service)

        self._metrics_service = metrics_service
        self._vocabulary_service = vocabulary_service
        self._process_service = process_service
        self._log_service = log_service

        self._device = arguments_service.device
        self._metric_types = arguments_service.metric_types

        pretrained_options = PretrainedRepresentationsOptions(
            include_pretrained_model=arguments_service.include_pretrained_model,
            pretrained_max_length=arguments_service.pretrained_max_length,
            pretrained_model_size=arguments_service.pretrained_model_size,
            pretrained_weights=arguments_service.pretrained_weights,
            pretrained_model=arguments_service.pretrained_model,
            fine_tune_pretrained=arguments_service.fine_tune_pretrained)

        self._shared_embedding_layer = None
        if self._arguments_service.share_embedding_layer:
            embedding_layer_options = EmbeddingLayerOptions(
                device=self._device,
                pretrained_representations_options=pretrained_options,
                learn_character_embeddings=arguments_service.learn_new_embeddings,
                vocabulary_size=vocabulary_service.vocabulary_size(),
                character_embeddings_size=arguments_service.encoder_embedding_size,
                dropout=arguments_service.dropout,
                output_embedding_type=EmbeddingType.Character)

            self._shared_embedding_layer = EmbeddingLayer(
                file_service,
                embedding_layer_options)

        self._encoder = SequenceEncoder(
            file_service=file_service,
            device=self._device,
            pretrained_representations_options=pretrained_options,
            embedding_size=arguments_service.encoder_embedding_size,
            input_size=vocabulary_service.vocabulary_size(),
            hidden_dimension=arguments_service.hidden_dimension,
            number_of_layers=arguments_service.number_of_layers,
            dropout=arguments_service.dropout,
            learn_embeddings=arguments_service.learn_new_embeddings,
            bidirectional=arguments_service.bidirectional,
            use_own_embeddings=(
                not self._arguments_service.share_embedding_layer),
            shared_embedding_layer=self._shared_embedding_layer)

        self._decoder = SequenceDecoder(
            file_service=file_service,
            device=self._device,
            embedding_size=arguments_service.decoder_embedding_size,
            output_dimension=vocabulary_service.vocabulary_size(),
            attention_dimension=arguments_service.hidden_dimension,
            encoder_hidden_dimension=arguments_service.hidden_dimension,
            decoder_hidden_dimension=arguments_service.hidden_dimension * 2,
            number_of_layers=arguments_service.number_of_layers,
            vocabulary_size=self._vocabulary_service.vocabulary_size(),
            pad_idx=self._vocabulary_service.pad_token,
            dropout=arguments_service.dropout,
            use_own_embeddings=(
                not self._arguments_service.share_embedding_layer),
            shared_embedding_layer=self._shared_embedding_layer,
            use_beam_search=self._arguments_service.use_beam_search,
            beam_width=self._arguments_service.beam_width,
            teacher_forcing_ratio=self._arguments_service.teacher_forcing_ratio,
            eos_token=self._vocabulary_service.eos_token)

        self._generator = SequenceGenerator(
            hidden_size=arguments_service.hidden_dimension,
            vocab_size=vocabulary_service.vocabulary_size())

        self.apply(self.init_weights)

        self._dev_edit_distances: List[float] = []
        self.metric_log_key = 'Levenshtein distance improvement (%)'

    @staticmethod
    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    @overrides
    def forward(self, input_batch: BatchRepresentation, debug=False, **kwargs):
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_hidden, encoder_final  = self._encoder.forward(input_batch)

        outputs = self._decoder.forward(
            input_batch.targets[:, :-1], # we remove the [EOS] tokens
            encoder_hidden,
            encoder_final,
            input_batch.generate_mask(input_batch.character_sequences))

        # return outputs, targets
        gen_outputs = self._generator.forward(outputs)

        return gen_outputs, input_batch.targets

    @overrides
    def calculate_accuracies(self, batch: BatchRepresentation, outputs, output_characters=False) -> Dict[MetricType, float]:
        output, targets = outputs
        output_dim = output.shape[-1]
        predictions = output.max(dim=2)[1].cpu().detach().numpy()

        targets = targets[:, 1:].cpu().detach().numpy()
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

        output_log = None
        if MetricType.LevenshteinDistance in self._metric_types:
            predicted_strings = [self._vocabulary_service.ids_to_string(
                x) for x in predicted_characters]
            target_strings = [self._vocabulary_service.ids_to_string(
                x) for x in target_characters]

            levenshtein_distance = np.mean([self._metrics_service.calculate_normalized_levenshtein_distance(
                predicted_strings[i], target_strings[i]) for i in range(len(predicted_strings))])

            metrics[MetricType.LevenshteinDistance] = levenshtein_distance

            if not self.training:
                predicted_levenshtein_distances = [
                    self._metrics_service.calculate_levenshtein_distance(
                        predicted_string, target_string) for predicted_string, target_string in zip(predicted_strings, target_strings)
                ]
                self._dev_edit_distances.extend(
                    predicted_levenshtein_distances)

            if output_characters:
                output_log = DataOutputLog()
                ocr_texts = batch.character_sequences.cpu().detach().tolist()
                for i in range(len(ocr_texts)):
                    input_string = self._vocabulary_service.ids_to_string(
                        ocr_texts[i])

                    output_log.add_new_data(
                        input_data=input_string,
                        output_data=predicted_strings[i],
                        true_data=target_strings[i])

        return metrics, output_log

    @overrides
    def optimizer_parameters(self):
        if not self._arguments_service.fine_tune_learning_rate:
            return self.parameters()

        embedding_layer = None
        if self._shared_embedding_layer is not None:
            embedding_layer = self._shared_embedding_layer
        else:
            embedding_layer = self._encoder._embedding_layer

        if not embedding_layer._include_pretrained:
            return self.parameters()

        pretrained_layer_parameters = embedding_layer._pretrained_layer.parameters()
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
    def calculate_evaluation_metrics(self) -> Dict[str, float]:
        if len(self._dev_edit_distances) == 0:
            return {}

        predicted_edit_sum = sum(self._dev_edit_distances)

        original_edit_sum = self._process_service.original_levenshtein_distance_sum

        improvement_percentage = (
            1 - (float(predicted_edit_sum) / original_edit_sum)) * 100

        result = {
            self.metric_log_key: improvement_percentage
        }

        return result

    @overrides
    def finalize_batch_evaluation(self, is_new_best: bool):
        if is_new_best:
            predicted_histogram = np.histogram(
                self._dev_edit_distances, bins=100)
            self._log_service.log_summary(
                'best-results-edit-distances-count', predicted_histogram[0])
            self._log_service.log_summary(
                'best-results-edit-distances-bins', predicted_histogram[1])

        self._dev_edit_distances = []