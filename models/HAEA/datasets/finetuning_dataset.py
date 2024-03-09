from torch.utils.data import Dataset
import torch
import math

class Seq2SeqDataset(Dataset):

    def __init__(self, source_dataset: torch.Tensor, time_len: int, n_only_input: int = 0):
        # dataset.shape = (time, var_len, hidden)
        self.source_dataset = source_dataset.unsqueeze(1)
        self.time_len = time_len
        self.n_only_input = n_only_input
        self.make_dataset()

    def __len__(self):
        return self.source_dataset.size(0)-(self.time_len + self.time_len+1)

    def get_data(self, indicate, source: bool = True):
        result = []
        for t in indicate:
            data = self.source_dataset[t]
            if not source and self.n_only_input > 0:
                data = data[:, :-self.n_only_input, :]
            result.append(data)

        # result.shape = (time_len * var, hidden)
        result = torch.concat(result, dim=0)
        return result

    def make_dataset(self):
        dataset_inc = []
        for t in range(self.time_len, self.source_dataset.size(0)-self.time_len-1):
            src = [r for r in range(t-self.time_len, t)]
            tgt = [r for r in range(t, t + self.time_len + 1)]
            dataset_inc.append((src, tgt))

        self.dataset_inc = dataset_inc


    def __getitem__(self, item):
        src_ind, tgt_ind = self.dataset_inc[item]
        src_ind = torch.tensor(src_ind)
        tgt_ind = torch.tensor(tgt_ind)

        src = self.get_data(src_ind)
        tgt = self.get_data(tgt_ind, source=False)

        return {
            "source": src,
            "source_id": src_ind,
            "target": tgt,
            "target_id": tgt_ind
        }


