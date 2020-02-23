import os
from services.arguments_service_base import ArgumentsServiceBase


class FileService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase):

        self._arguments_service = arguments_service

    def get_data_path(self) -> str:
        data_path = self._arguments_service.get_argument('data_folder')

        if not os.path.exists(data_path):
            os.mkdir(data_path)

        challenge_name = self._arguments_service.get_argument('challenge')
        if challenge_name:
            data_path = os.path.join(data_path, challenge_name)
            if not os.path.exists(data_path):
                os.mkdir(data_path)

        data_model_path = os.path.join(
            data_path,
            str(self._arguments_service.get_argument('configuration')))

        if not os.path.exists(data_model_path):
            os.mkdir(data_model_path)

        data_language_path = os.path.join(
            data_model_path,
            str(self._arguments_service.get_argument('language')))

        if not os.path.exists(data_language_path):
            os.mkdir(data_language_path)

        return data_language_path

    def get_checkpoints_path(self) -> str:
        if not self._arguments_service.get_argument('checkpoint_folder'):
            output_path = self._arguments_service.get_argument('output_folder')
        else:
            output_path = self._arguments_service.get_argument(
                'checkpoint_folder')

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        challenge_name = self._arguments_service.get_argument('challenge')
        if challenge_name:
            output_path = os.path.join(output_path, challenge_name)
            if not os.path.exists(output_path):
                os.mkdir(output_path)

        output_model_path = os.path.join(
            output_path,
            str(self._arguments_service.get_argument('configuration')))

        if not os.path.exists(output_model_path):
            os.mkdir(output_model_path)

        model_path = os.path.join(
            output_model_path,
            str(self._arguments_service.get_argument('language')))

        if not os.path.exists(model_path):
            os.mkdir(model_path)

        return model_path

    def get_pickles_path(self) -> str:
        data_path = self._arguments_service.get_argument('data_folder')

        if not os.path.exists(data_path):
            os.mkdir(data_path)

        challenge_name = self._arguments_service.get_argument('challenge')
        if challenge_name:
            data_path = os.path.join(data_path, challenge_name)
            if not os.path.exists(data_path):
                os.mkdir(data_path)

        data_pickles_path = os.path.join(data_path, 'pickles')

        if not os.path.exists(data_pickles_path):
            os.mkdir(data_pickles_path)

        return data_pickles_path
