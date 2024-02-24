from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):

    def __init__(self, source_dataset: torch.Tensor, label_dataset: torch.Tensor, max_lead_time: int = 78):
        # dataset.shape = (time, var_len, hidden)
        self.source_dataset = source_dataset
        self.label_dataset = label_dataset
        self.max_lead_time = max_lead_time

    def __len__(self):
        return self.source_dataset.size(0)

    def get_data(self, t):
        range = min(t + self.max_lead_time, self.source_dataset.size(0)-1)
        delta = range-t
        sample = torch.randint(t, range, (1,))
        return self.source_dataset[t], self.label_dataset[sample]

    def __getitem__(self, t):
        src_ind, tgt_ind = self.dataset_inc[item]
        src = self.get_data(src_ind, self.source_dataset)
        tgt = self.get_data(tgt_ind, self.source_dataset, source=False)
        label = self.get_data(tgt_ind, self.label_dataset, source=False)
        return src, tgt, label

