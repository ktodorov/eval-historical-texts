from enums.run_type import RunType

from datasets.dataset_base import DatasetBase

class DatasetService:
    def __init__(self):
        pass

    def get_dataset(self, run_type: RunType) -> DatasetBase:
        result = DatasetBase()
        return result