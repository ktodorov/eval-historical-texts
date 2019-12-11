import torch
from torch import nn
import random

from entities.model_checkpoint import ModelCheckpoint
from models.model_base import ModelBase

from services.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService

from models.multifit_encoder import MultiFitEncoder
from models.multifit_decoder import MultiFitDecoder


class MultiFitModel(ModelBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            data_service: DataService):
        super(MultiFitModel, self).__init__(data_service, arguments_service)

        self._device = arguments_service.get_argument('device')

        self._output_dimension = self._arguments_service.get_argument(
            'sentence_piece_vocabulary_size')

        self._encoder = MultiFitEncoder(
            embedding_size=self._arguments_service.get_argument(
                'embedding_size'),
            input_size=self._arguments_service.get_argument(
                'sentence_piece_vocabulary_size'),
            hidden_dimension=self._arguments_service.get_argument(
                'hidden_dimension'),
            number_of_layers=self._arguments_service.get_argument(
                'number_of_layers'),
            dropout=self._arguments_service.get_argument('dropout')
        )

        self._decoder = MultiFitDecoder(
            embedding_size=self._arguments_service.get_argument(
                'embedding_size'),
            output_dimension=self._output_dimension,
            hidden_dimension=self._arguments_service.get_argument(
                'hidden_dimension'),
            number_of_layers=self._arguments_service.get_argument(
                'number_of_layers'),
            dropout=self._arguments_service.get_argument('dropout')
        )

        self.apply(self.init_weights)

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
        hidden, cell = self._encoder.forward(source, lengths)

        # first input to the decoder is the <sos> tokens
        input = targets[:, 0]
        teacher_forcing_ratio = 0.5

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self._decoder.forward(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
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

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self._device)

    def calculate_accuracy(self, batch, outputs) -> bool:
        # targets = batch[2]
        return 0

    def compare_metric(self, best_metric, metrics) -> bool:
        if best_metric is None or best_metric > metrics:
            return True
