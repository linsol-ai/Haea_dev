from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):

    def __init__(self, source_dataset: torch.Tensor, time_len: int, max_lead_time: int = 78):
        # dataset.shape = (time, var_len, hidden)
        self.source_dataset = source_dataset
        self.max_lead_time = max_lead_time
        self.time_len = time_len
        self.sample = torch.arange(time_len, max_lead_time, step=time_len, dtype=torch.int)


    def __len__(self):
        return self.source_dataset.size(0)


    def __getitem__(self, t):
        t = max(t, self.time_len-1)
        delta = self.sample[torch.randint(0, self.sample.size(0)-1, (1,))].item()
        t = t - max(0, (t + delta + self.time_len) - (self.source_dataset.size(0) - 1))
        src = self.source_dataset[t-self.time_len+1:t+1]
        next = t + delta + 1
        tgt = self.source_dataset[next:next+self.time_len]
        return src, tgt, delta
        

