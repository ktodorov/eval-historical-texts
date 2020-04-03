import torch
from torch import nn
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
from services.vocabulary_service import VocabularyService
from services.pretrained_representations_service import PretrainedRepresentationsService
from services.decoding_service import DecodingService

from models.rnn_encoder_decoder.sequence_encoder import SequenceEncoder
from models.rnn_encoder_decoder.sequence_decoder import SequenceDecoder

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class SequenceModel(ModelBase):
    def __init__(
            self,
            arguments_service: PostOCRArgumentsService,
            data_service: DataService,
            metrics_service: MetricsService,
            vocabulary_service: VocabularyService,
            pretrained_representations_service: PretrainedRepresentationsService,
            decoding_service: DecodingService):
        super(SequenceModel, self).__init__(data_service, arguments_service)

        self._metrics_service = metrics_service
        self._vocabulary_service = vocabulary_service
        self._decoding_service = decoding_service

        self._device = arguments_service.device
        self._metric_types = arguments_service.metric_types

        self._shared_embeddings = None
        if self._arguments_service.share_embedding_layer:
            self._shared_embeddings = nn.Embedding(
                vocabulary_service.vocabulary_size(),
                self._arguments_service.encoder_embedding_size)

        self._encoder = SequenceEncoder(
            pretrained_representations_service=pretrained_representations_service,
            embedding_size=arguments_service.encoder_embedding_size,
            input_size=vocabulary_service.vocabulary_size(),
            hidden_dimension=arguments_service.hidden_dimension,
            number_of_layers=arguments_service.number_of_layers,
            dropout=arguments_service.dropout,
            include_pretrained=arguments_service.include_pretrained_model,
            pretrained_hidden_size=arguments_service.pretrained_model_size,
            learn_embeddings=arguments_service.learn_new_embeddings,
            bidirectional=True,
            use_own_embeddings=(
                not self._arguments_service.share_embedding_layer),
            shared_embeddings=(lambda x: self._shared_embeddings(x))
        )

        self._decoder = SequenceDecoder(
            embedding_size=arguments_service.decoder_embedding_size,
            output_dimension=vocabulary_service.vocabulary_size(),
            hidden_dimension=arguments_service.hidden_dimension * 2,
            number_of_layers=arguments_service.number_of_layers,
            dropout=arguments_service.dropout,
            use_own_embeddings=(
                not self._arguments_service.share_embedding_layer),
            shared_embeddings=(lambda x: self._shared_embeddings(x))
        )

        self.apply(self.init_weights)

    def init_hidden(self, batch_size, hidden_dimension):
        # initialize the hidden state and the cell state to zeros
        return (torch.zeros(batch_size, hidden_dimension).to(self._device),
                torch.zeros(batch_size, hidden_dimension).to(self._device))

    @staticmethod
    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, input_batch, debug=False, **kwargs):
        source, targets, lengths, pretrained_representations, offset_lists = input_batch

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_context = self._encoder.forward(
            source, lengths, pretrained_representations, offset_lists, debug=debug)

        hidden = encoder_context
        encoder_context = encoder_context.permute(1, 0, 2)

        if self._arguments_service.use_beam_search:
            outputs, targets = self._decoding_service.beam_decode(
                targets,
                encoder_context,
                (lambda x, y, z: self._decoder.forward(x, y, z)))
        else:
            outputs, targets = self._decoding_service.greedy_decode(
                targets,
                encoder_context,
                (lambda x, y, z: self._decoder.forward(x, y, z)))

        if outputs.shape[1] < targets.shape[1]:
            padded_output = torch.zeros((outputs.shape[0], targets.shape[1], outputs.shape[2])).to(
                self._arguments_service.device)
            padded_output[:, :output.shape[1], :] = outputs
            outputs = padded_output
        elif outputs.shape[1] > targets.shape[1]:
            padded_targets = torch.zeros((targets.shape[0], outputs.shape[1]), dtype=torch.int64).to(
                self._arguments_service.device)
            padded_targets[:, :targets.shape[1]] = targets
            targets = padded_targets

        return outputs, targets

    def calculate_accuracies(self, batch, outputs, output_characters=False) -> Dict[MetricType, float]:
        output, targets = outputs
        output_dim = output.shape[-1]
        predictions = output.max(dim=2)[1].cpu().detach().numpy()

        targets = targets.cpu().detach().numpy()
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
                ocr_texts_tensor, _, lengths, _, _ = batch
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

        result = best_metric.get_current_loss() > new_metric.get_current_loss()
        return result
