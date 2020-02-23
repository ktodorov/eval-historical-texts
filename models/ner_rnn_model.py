import torch
from torch import nn

from models.model_base import ModelBase

from services.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService

class NERRNNModel(ModelBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            data_service: DataService):
        super().__init__(data_service)

        vocab_size = 200
        embedding_dim = 100
        lstm_hidden_dim = 100
        number_of_tags = 5


        #maps each token to an embedding_dim vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        #the LSTM takens embedded sentence
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True)

        #fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(lstm_hidden_dim, number_of_tags)