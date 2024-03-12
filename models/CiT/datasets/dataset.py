from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):

    def __init__(self, source_dataset: torch.Tensor, var_seq: torch.Tensor, time_len: int, max_lead_time: int = 78):
        # dataset.shape = (time, var_len, hidden)
        self.source_dataset = source_dataset
        self.max_lead_time = max_lead_time
        self.time_len = time_len
        self.var_seq = var_seq
        
        self.make_dataset()


    def __len__(self):
        return self.source_dataset.size(0)-self.time_len


    def get_data(self, item):
        sample = torch.randint(t, next, (1,)).item()
        t = item + self.time_len
        src = self.source_dataset[t-self.time_len+1:t+1]

        return src, tgt


    def make_dataset(self):
        dataset_inc = []
        for t in range(self.time_len, self.source_dataset.size(0)):
            src = [r for r in range(t-self.time_len+1, t+1)]
            next = min(t + self.max_lead_time, self.source_dataset.size(0)-1)
            if t != next:
                sample = torch.randint(t, next, (1,)).item()
            else:
                sample = t
            delta = sample-t
           
            dataset_inc.append((src, sample, delta))

        self.dataset_inc = dataset_inc

    def __getitem__(self, item):
        src_ind, sample, delta = self.dataset_inc[item]
        src, tgt = self.get_data(src_ind, sample)
        return src, tgt, delta, self.var_seq
        

