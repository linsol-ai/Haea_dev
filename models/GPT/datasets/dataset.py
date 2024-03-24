from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):

    def __init__(self, source_dataset: torch.Tensor, time_len: int):
        # dataset.shape = (time, var_len, hidden)
        self.source_dataset = source_dataset
        self.time_len = time_len

    def __len__(self):
        return self.source_dataset.size(0)

    def __getitem__(self, t):
        t = max(t, 1)
        t = min(t + self.time_len + 1, self.source_dataset.size(0) - 1) - (self.time_len + 1)
        src = self.source_dataset[t:t + self.time_len + 2]
        return src
        

class ValidationDataset(Dataset):
    def __init__(self, source_dataset: torch.Tensor, time_len: int, max_lead_time: int = 78):
        # dataset.shape = (time, var_len, hidden)
        self.source_dataset = source_dataset
        self.max_lead_time = max_lead_time
        self.time_len = time_len


    def __len__(self):
        return self.source_dataset.size(0)


    def __getitem__(self, day):
        src_st, src_ed = (day) * self.time_len, (day+1) * self.time_len
        #src.shape (time, var_len, hidden)
        src = self.source_dataset[src_st:src_ed]
        tgt = self.source_dataset[src_ed:src_ed + self.ma]

        return src, tgt, self.sample
        

