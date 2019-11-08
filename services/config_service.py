import os

class ConfigService:
    _directory_config_key: str = 'directory'
    _directory: str = ''

    def __init__(self, config: dict):
        self._validate_configuration(config)

        self._directory = os.path.join('.', config[self._directory_config_key])

    def _validate_configuration(self, config: dict):
        if self._directory_config_key not in config.keys():
            raise Exception(f'No configuration available for "{self._directory_config_key}"')

    @property
    def directory(self) -> str:
        return self._directory
