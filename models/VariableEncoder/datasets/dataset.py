from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):

    def __init__(self, source_dataset: torch.Tensor, label_dataset: torch.Tensor, tgt_time_len: int, ):
        # dataset.shape = (time, var_len, hidden)
        self.source_dataset = source_dataset.unsqueeze(1)
        self.label_dataset = label_dataset.unsqueeze(1)
        self.tgt_time_len = tgt_time_len
        self.make_dataset()

    def __len__(self):
        return self.source_dataset.size(0)-(self.tgt_time_len)

    def get_data(self, indicate, dataset):
        result = []
        for t in indicate:
            result.append(dataset[t])

        # dataset.shape = (time_len, var, hidden)
        result = torch.concat(result, dim=0)
        return result

    def make_dataset(self):
        dataset_inc = []
        for t in range(self.source_dataset.size(0)-self.tgt_time_len):
            src = [t]
            tgt = range(t+1, t+1 + self.tgt_time_len)
            dataset_inc.append((src, tgt))

        self.dataset_inc = dataset_inc

    def __getitem__(self, item):
        src_ind, tgt_ind = self.dataset_inc[item]
        src = self.get_data(src_ind, self.source_dataset)
        tgt = self.get_data(tgt_ind, self.source_dataset)
        label = self.get_data(tgt_ind, self.label_dataset)
        return src, tgt, label

