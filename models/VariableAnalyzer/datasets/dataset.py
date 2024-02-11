from torch.utils.data import Dataset
import torch
import random
import numpy as np
import time

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
        start = time.time()
        dataset = []
        for t in indicate:
            if t >= self.dataset.size(0):
                dataset.append(torch.zeros_like(self.dataset[0]))
            else:
                dataset.append(self.dataset[t])

        # dataset.shape = (time_len, var, hidden)
        dataset = torch.stack(dataset, dim=0)
        dataset = dataset.view(-1, dataset.size(2))
        end = time.time()
        print('getdata: ', f"{end - start:.5f} sec")
        return dataset

    def make_dataset(self):
        dataset_inc = []
        for t in range(self.dataset.size(0)):
            src = [t]
            tgt = range(t+1, t+1 + self.time_len)
            dataset_inc.append((src, tgt))

        self.dataset_inc = dataset_inc


    def __getitem__(self, item):
        src, tgt = self.dataset_inc[item]
        src = self.get_data(src)
        tgt = self.get_data(tgt)
        start = time.time()
        res = torch.concat([src, tgt], dim=0)
        end = time.time()
        print('res: ', f"{end - start:.5f} sec")
        return res

