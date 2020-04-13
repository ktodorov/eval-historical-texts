from torch.utils.data import Dataset


class DatasetBase(Dataset):
    def __init__(self, **kwargs):
        super().__init__()

    def __len__(self):
        return super().__len__()

    def __getitem__(self):
        return super().__getitem__()

    def use_collate_function(self) -> bool:
        return False

    def collate_function(self, sequences):
        pass
