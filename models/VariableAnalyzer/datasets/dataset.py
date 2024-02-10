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
        self.make_dataset()

    def __len__(self):
        return self.dataset.size(0)

    def get_data(self, indicate):
        dataset = []
        for t in indicate:
            if t >= self.dataset.size(0):
                dataset.append(torch.zeros_like(self.dataset[0]))
            else:
                dataset.append(self.dataset[t])

        # dataset.shape = (time_len, var, hidden)
        dataset = torch.stack(dataset, dim=0)
        dataset = dataset.view(-1, dataset.size(2))
        return dataset

    def make_dataset(self):
        dataset = []
        for t in range(self.dataset.size(0)):
            src = [t]
            tgt = [range(t+1, t+1 + self.time_len)] 
            dataset.append((src, tgt))

        self.dataset_idx = dataset


    def __getitem__(self, item):
        return self.dataset[item]

