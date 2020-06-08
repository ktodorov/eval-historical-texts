from enums.configuration import Configuration

from models.model_base import ModelBase
from models.transformer_fine_tune.kbert_model import KBertModel
from models.transformer_fine_tune.kxlnet_model import KXLNetModel
from models.word2vec.cbow_model import CBOWModel

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService
from services.vocabulary_service import VocabularyService
from services.process.process_service_base import ProcessServiceBase
from services.file_service import FileService

class ModelService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            data_service: DataService,
            vocabulary_service: VocabularyService,
            process_service: ProcessServiceBase,
            file_service: FileService):
        self._arguments_service = arguments_service
        self._data_service = data_service
        self._vocabulary_service = vocabulary_service
        self._process_service = process_service
        self._file_service = file_service

    def create_model(self) -> ModelBase:
        configuration: Configuration = self._arguments_service.configuration

        device = self._arguments_service.device

        if configuration == Configuration.KBert:
            return KBertModel(self._arguments_service, self._data_service, output_hidden_states=True).to(device)
        elif configuration == Configuration.XLNet:
            return KXLNetModel(self._arguments_service, self._data_service).to(device)
        elif configuration == Configuration.CBOW:
            return CBOWModel(
                arguments_service=self._arguments_service,
                vocabulary_service=self._vocabulary_service,
                data_service=self._data_service,
                process_service=self._process_service,
                file_service=self._file_service,
                use_only_embeddings=(self._arguments_service.evaluate or self._arguments_service.run_experiments)).to(device)

        raise LookupError(f'The {str(configuration)} is not supported')
