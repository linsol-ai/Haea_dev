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
        t = min(t + self.time_len + 2, self.source_dataset.size(0) - 1) - (self.time_len + 2)
        tgt = self.source_dataset[next:next+self.time_len]
        return src, tgt, delta
        

class ValidationDataset(Dataset):
    def __init__(self, source_dataset: torch.Tensor, time_len: int, max_lead_time: int = 78):
        # dataset.shape = (time, var_len, hidden)
        self.source_dataset = source_dataset
        self.max_lead_time = max_lead_time
        self.time_len = time_len
        self.sample = torch.arange(0, max_lead_time, step=time_len, dtype=torch.int32)


    def __len__(self):
        return (self.source_dataset.size(0) // 24) - (self.sample.size(0) + 1)


    def __getitem__(self, day):
        src_st, src_ed = (day) * self.time_len, (day+1) * self.time_len
        tgt_st = self.sample + src_ed
        tgt_ed = tgt_st + self.time_len
        #src.shape (lead_days, time, var_len, hidden)
        src = self.source_dataset[src_st:src_ed].unsqueeze(0).repeat_interleave(repeats=self.sample.size(0), dim=0)
        tgt = []
        for i in range(self.sample.size(0)):
            tgt.append(self.source_dataset[tgt_st[i]:tgt_ed[i]].unsqueeze(0))
        tgt = torch.cat(tgt, dim=0)

        return src, tgt, self.sample
        

