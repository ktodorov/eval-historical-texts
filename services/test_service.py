from models.model_base import ModelBase

from entities.model_checkpoint import ModelCheckpoint

from enums.evaluation_type import EvaluationType

from services.arguments_service_base import ArgumentsServiceBase
from services.dataloader_service import DataLoaderService
from services.evaluation_service import EvaluationService

from utils.dict_utils import update_dictionaries


class TestService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            dataloader_service: DataLoaderService,
            evaluation_service: EvaluationService,
            model: ModelBase):

        self._arguments_service = arguments_service
        self._evaluation_service = evaluation_service

        self._model = model.to(arguments_service.get_argument('device'))
        self._load_model()
        self._model.eval()

        self._dataloader = dataloader_service.get_test_dataloader()

    def test(self) -> bool:
        evaluation: Dict[EvaluationType, List] = {}

        for i, batch in enumerate(self._dataloader):
            batch = batch.unsqueeze(1).to(
                self._arguments_service.get_argument('device'))
            outputs = self._model.forward(batch)

            batch_evaluation = self._evaluation_service.evaluate(
                outputs,
                self._arguments_service.get_argument('evaluation_type')
            )

            update_dictionaries(evaluation, batch_evaluation)

        print(evaluation)
        return True

    def _load_model(self) -> ModelCheckpoint:
        checkpoints_path = self._get_checkpoints_path()
        model_checkpoint = self._model.load(checkpoints_path, 'BEST')
        return model_checkpoint

    def _get_checkpoints_path(self) -> str:
        if not self._arguments_service.get_argument('checkpoint_folder'):
            return self._arguments_service.get_argument('output_folder')

        return self._arguments_service.get_argument('checkpoint_folder')
