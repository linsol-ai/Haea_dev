from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):

    def __init__(self, source_dataset: torch.Tensor, label_dataset: torch.Tensor, max_lead_time: int = 78):
        # dataset.shape = (time, 1, var_len, hidden)
        self.source_dataset = source_dataset.unsqueeze(1)
        self.label_dataset = label_dataset.unsqueeze(1)
        self.max_lead_time = max_lead_time
        self.make_dataset()

    def __len__(self):
        return self.source_dataset.size(0)

    def get_data(self, indicate, dataset, source: bool = True):
        result = []
        for t in indicate:
            data = dataset[t]
            if not source:
                data = data[:, :-self.n_only_input, :]
            result.append(data)

        # dataset.shape = (time_len, var, hidden)
        result = torch.concat(result, dim=0)
        return result

    def make_dataset(self):
        dataset_inc = []
        for t in range(self.source_dataset.size(0)-self.tgt_time_len):
            src = [t]
            tgt = range(t, t + self.tgt_time_len)
            dataset_inc.append((src, tgt))

        self.dataset_inc = dataset_inc

    def __getitem__(self, item):
        src_ind, tgt_ind = self.dataset_inc[item]
        src = self.get_data(src_ind, self.source_dataset)
        tgt = self.get_data(tgt_ind, self.source_dataset, source=False)
        label = self.get_data(tgt_ind, self.label_dataset, source=False)
        return src, tgt, label

