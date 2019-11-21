from models.model_base import ModelBase

from entities.model_checkpoint import ModelCheckpoint

from services.arguments_service_base import ArgumentsServiceBase
from services.dataloader_service import DataLoaderService


class TestService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            dataloader_service: DataLoaderService,
            model: ModelBase):

        self._arguments_service = arguments_service
        self._model = model.to(arguments_service.get_argument('device'))
        self._load_model()
        self._model.eval()

        self._dataloader = dataloader_service.get_test_dataloader()

    def test(self) -> bool:
        for i, batch in enumerate(self._dataloader):
            batch = batch.unsqueeze(1).to(self._arguments_service.get_argument('device'))
            outputs = self._model.forward(batch)

            print(batch)

            # TODO Do something with the outputs...

    def _load_model(self) -> ModelCheckpoint:
        checkpoints_path = self._get_checkpoints_path()
        model_checkpoint = self._model.load(checkpoints_path, 'BEST')
        return model_checkpoint

    def _get_checkpoints_path(self) -> str:
        if not self._arguments_service.get_argument('checkpoint_folder'):
            return self._arguments_service.get_argument('output_folder')

        return self._arguments_service.get_argument('checkpoint_folder')