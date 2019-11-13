from torch.utils.data import Dataset

class DatasetBase(Dataset):
    def __init__(self, **kwargs):
        super(DatasetBase, self).__init__()
