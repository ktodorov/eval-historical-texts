import os

from datasets.dataset_base import DatasetBase
from enums.run_type import RunType
from enums.language import Language
from preprocessing.ner_preprocessing import preprocess_data

from services.file_service import FileService
from services.tokenizer_service import TokenizerService


class NERDataset(DatasetBase):
    def __init__(
            self,
            file_service: FileService,
            tokenizer_service: TokenizerService,
            run_type: RunType,
            language: Language):
        super().__init__()

        data_path = file_service.get_data_path()
        file_suffix = 'train' if run_type == RunType.Train else 'dev'
        language_suffix = self._get_language_suffix(language)
        file_path = os.path.join(data_path, f'HIPE-data-v0.9-{file_suffix}-{language_suffix}.tsv')
        self.ne_collection = preprocess_data(file_path, tokenizer_service)

        raise Exception('test')


    def __len__(self):
        return len(self.ne_collection)

    def __getitem__(self, idx):
        result = self.ne_collection[idx]
        return result

    def _get_language_suffix(self, language: Language):
        if language == Language.English:
            return 'en'
        elif language == Language.French:
            return 'fr'
        elif language == Language.German:
            return 'de'
        else:
            raise Exception('Unsupported language')

    def use_collate_function(self) -> bool:
        return False
