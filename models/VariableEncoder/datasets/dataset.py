from torch.utils.data import Dataset
import torch
import math

class CustomDataset(Dataset):

    def __init__(self, source_dataset: torch.Tensor, src_time_len: int, tgt_time_len: int, n_only_input: int = 0):
        # dataset.shape = (time, 1, var_len, hidden)
        self.source_dataset = source_dataset.unsqueeze(1)
        self.src_time_len = src_time_len
        self.tgt_time_len = tgt_time_len
        self.n_only_input = n_only_input
        self.make_dataset()

    def __len__(self):
        return self.source_dataset.size(0)-(self.tgt_time_len)-1

    def get_data(self, indicate, source: bool = True):
        result = []
        for i, t in enumerate(indicate):
            data = self.source_dataset[t]
            if not source and self.n_only_input > 0:
                data = data[:, :-self.n_only_input, :]
            result.append(data)

        # result.shape = (time_len * var, hidden)
        result = torch.concat(result, dim=0)
        return result

    def make_dataset(self):
        dataset_inc = []
        for t in range(self.source_dataset.size(0)-self.tgt_time_len-1):
            src = [r for r in range(t-self.src_time_len, t+1)]
            tgt = [r for r in range(t+1, t+1 + self.tgt_time_len)]
            dataset_inc.append((src, tgt))

        self.dataset_inc = dataset_inc


    def __getitem__(self, item):
        src_ind, tgt_ind = self.dataset_inc[item]
        src = self.get_data(src_ind)
        tgt = self.get_data(tgt_ind, source=False)
        return src, tgt

