import torch
from torch.utils.data import Dataset
import pandas as pd

class DLDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir


    def __len__(self):
        return 1

    def __getitem__(self, item):
        """ get the clip indexed by the (idx) """
        idx_path = self.idx_path_list[item]



