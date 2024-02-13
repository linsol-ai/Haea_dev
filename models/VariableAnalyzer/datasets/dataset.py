from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):

    def __init__(self, input_dataset: torch.Tensor, tar_dataset: torch.Tensor, src_time_len: int, tgt_time_len: int):
        # dataset.shape = (time, var_len, hidden)
        self.input_dataset = input_dataset
        self.tar_dataset = tar_dataset
        self.var_len = input_dataset.size(1)
        self.src_time_len = src_time_len
        self.tgt_time_len = tgt_time_len
        self.make_dataset()


    def __len__(self):
        return self.input_dataset.size(0)-(self.src_time_len+self.tgt_time_len)

    def get_data(self, indicate, dataset):
        result = []
        for t in indicate:
            result.append(dataset[t])

        # dataset.shape = (time_len, var, hidden)
        result = torch.stack(result, dim=0)
        return result

    def make_dataset(self):
        dataset_inc = []
        for t in range(self.input_dataset.size(0)-self.time_len):
            src = [t]
            tgt = range(t+1, t+1 + self.time_len)
            dataset_inc.append((src, tgt))

        self.dataset_inc = dataset_inc


    def __getitem__(self, item):
        src_ind, tgt_ind = self.dataset_inc[item]
        src = self.get_data(src_ind, self.input_dataset)
        tgt = self.get_data(tgt_ind, self.input_dataset)
        label = self.get_data(tgt_ind, self.tar_dataset)
        return src, tgt, label

