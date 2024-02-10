from torch.utils.data import Dataset
import torch
import random
import numpy as np


class CustomDataset(Dataset):

    def __init__(self, dataset: torch.Tensor, time_len: int):
        # dataset.shape = (time, var_len, hidden)
        self.dataset = dataset
        self.var_len = dataset.size(1)
        self.time_len = time_len
        self.padding_vec = torch.
        self.make_dataset()

    def __len__(self):
        return self.dataset.size(0)

    def make_dataset(self):
        dataset = []
        for t in range(self.dataset.size(0)):
            src = [t]
            tgt = [range(t, t + self.time_len)] 
            dataset.append((src, tgt))
        self.dataset = dataset


    def __getitem__(self, item):
        return self.dataset[item]

