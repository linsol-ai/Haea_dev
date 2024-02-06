from torch.utils.data import Dataset
import torch
import random
import numpy as np


class HAEADataset(Dataset):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3

    special_tokens = [0, 1, 2, 3, 4]


    def __init__(self, dataset: torch.Tensor, time_len: int):
        # dataset.shape = (time, var_len, hidden)
        self.dataset = dataset
        self.var_len = dataset.size(1)
        self.time_len = time_len
        self.make_dataset()

    def __len__(self):
        return self.dataset.size(0)


    def make_dataset(self):
        dataset = []
        for t in range(self.dataset.size(0)):
            src = [(t, v) for v in range(self.var_len)]
            tgt = []
            for t_n in range(t, min(self.dataset.size(0), t + self.time_len)):
                tgt.extend(
                    [(t_n, v) for v in range(self.var_len)]
                )
            dataset.append((src, tgt))
        self.dataset = dataset


    def __getitem__(self, item):
        return self.dataset[item]

