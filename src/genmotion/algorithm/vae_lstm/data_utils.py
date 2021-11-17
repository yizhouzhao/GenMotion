import os
import numpy as np
import torch
from typing import Callable, Optional
from torch.utils.data import Dataset

class MotionDataset(Dataset):
    """Motion dataset
    :param root_dir: directory with all the motion npz files.
    :type root_dir: str
    :param fetch: optional fetch function to specify how data is fetched
    :type fetch: callable
    :param transform: optional transform function to be applied on a sample.
    :type transform: callable
    """

    def __init__(self, root_dir: str, fetch: Optional[Callable] = None, transform: Optional[Callable] = None):

        self.root_dir = root_dir
        self.transform = transform

        # get all file name
        if fetch:
            self.file_name = fetch(root_dir)
        else:
            self.file_name = []
            for r, d, f in os.walk(self.root_dir):
                for file in f:
                    self.file_name.append(r + "/" + file)

        self.sanity_check()                

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

            sample = []
            for i in idx: 
                if self.transform:
                    sample.append(self.transform(np.load(self.file_name[i]), allow_pickle=True))
            
        else:
            sample = np.load(self.file_name[idx], allow_pickle=True)
            if self.transform:
                sample = self.transform(sample)

        return sample

    def sanity_check(self):
        """sanity check of dataset. Need to be overloaded for custom dataset"""
        for file_path in self.file_name:
            data = np.load(file_path)
            for field in ["trans", "root_orient", "poses"]:
                if field not in data:
                    self.file_name.remove(file_path)
                    break