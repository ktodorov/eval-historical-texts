from torch.utils.data import Dataset

class DatasetBase(Dataset):
    def __init__(self, **kwargs):
        super(DatasetBase, self).__init__()

    def use_collate_function(self) -> bool:
        return False

    def collate_function(self, sequences):
        pass