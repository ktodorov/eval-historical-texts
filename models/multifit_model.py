import torch
from torch import nn
import random

from entities.model_checkpoint import ModelCheckpoint
from models.model_base import ModelBase

from services.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService

from models.encoder import Encoder
from models.decoder import Decoder


class MultiFitModel(ModelBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            data_service: DataService):
        super(MultiFitModel, self).__init__(data_service, arguments_service)

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input_batch, **kwargs):
        source, targets, lengths = input_batch
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        (batch_size, trg_len) = targets.shape
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len,
                              trg_vocab_size).to('cuda')

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(source)

        # first input to the decoder is the <sos> tokens
        input = targets[:, 0]
        teacher_forcing_ratio = 0.5

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[:, t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = targets[:, t] if teacher_force else top1

        return outputs, targets, lengths

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device='cuda')

    def calculate_accuracy(self, predictions, targets) -> int:
        return 0

    def compare_metric(self, best_metric, metrics) -> bool:
        if best_metric is None or best_metric > metrics:
            return True
