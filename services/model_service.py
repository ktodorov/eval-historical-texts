from enums.configuration import Configuration

from models.model_base import ModelBase
from models.kbert_model import KBertModel
from models.kxlnet_model import KXLNetModel

from services.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService


class ModelService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            data_service=DataService):
        self._arguments_service = arguments_service
        self._data_service = data_service

    def create_model(self) -> ModelBase:
        configuration: Configuration = self._arguments_service.configuration

        device = self._arguments_service.device

        if configuration == Configuration.KBert:
            return KBertModel(self._arguments_service, self._data_service).to(device)
        elif configuration == Configuration.XLNet:
            return KXLNetModel(self._arguments_service, self._data_service).to(device)

        raise LookupError(f'The {str(configuration)} is not supported')
