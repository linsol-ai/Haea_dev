from torch.utils.data import Dataset
import torch
import random
import numpy as np
import time

class CustomDataset(Dataset):

    def __init__(self, input_dataset: torch.Tensor, tar_dataset: torch.Tensor, time_len: int, device):
        # dataset.shape = (time, var_len, hidden)
        self.input_dataset = input_dataset
        self.tar_dataset = tar_dataset
        self.var_len = input_dataset.size(1)
        self.time_len = time_len
        self.device = device
        self.make_dataset()

    def __len__(self):
        return self.input_dataset.size(0)-self.time_len

    def get_data(self, indicate, dataset):
        result = []
        for t in indicate:
            result.append(dataset[t].to(self.device))

        # dataset.shape = (time_len, var, hidden)
        result = torch.stack(result, dim=0)
        result = result.view(-1, result.size(2))
        return result

    def make_dataset(self):
        dataset_inc = []
        for t in range(self.dataset.size(0)-self.time_len):
            src = [t]
            tgt = range(t+1, t+1 + self.time_len)
            dataset_inc.append((src, tgt))

        self.dataset_inc = dataset_inc


    def __getitem__(self, item):
        src, tgt = self.dataset_inc[item]
        src = self.get_data(src)
        tgt = self.get_data(tgt)
        
        res = torch.concat([src, tgt], dim=0)
        return res.cpu()

