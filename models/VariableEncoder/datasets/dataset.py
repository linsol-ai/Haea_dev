from torch.utils.data import Dataset
import torch
import math

class CustomDataset(Dataset):

    SPECIAL_TOKEN_BOS = 0
    SPECIAL_TOKEN_EOS = 1

    def __init__(self, source_dataset: torch.Tensor, tgt_time_len: int, n_only_input: int = 0):
        # dataset.shape = (time, var_len, hidden)
        self.source_dataset = source_dataset
        self.tgt_time_len = tgt_time_len
        self.n_only_input = n_only_input
        self.pe = self.positional_encoding(source_dataset.size(-1), tgt_time_len + 10)
        self.make_dataset()

    def __len__(self):
        return self.source_dataset.size(0)-(self.tgt_time_len)

    def get_data(self, indicate, source: bool = True):
        result = []
        for i, t in enumerate(indicate):
            if t == self.SPECIAL_TOKEN_BOS or t == self.SPECIAL_TOKEN_EOS:
                data = torch.zeros(1, self.source_dataset.size(-1))
            else:  
                data = self.source_dataset[t]
                if not source and self.n_only_input > 0:
                    data = data[:, :-self.n_only_input, :]
            data = data + self.pe[i].unsqueeze(dim=0).repeat_interleave(data.size(0), dim=0)
            result.append(data)

        # result.shape = (time_len * var + 2, hidden)
        result = torch.concat(result, dim=0)
        return result

    def positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model).float()

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe


    def make_dataset(self):
        dataset_inc = []
        for t in range(self.source_dataset.size(0)-self.tgt_time_len):
            src = [self.SPECIAL_TOKEN_BOS, t+2, self.SPECIAL_TOKEN_EOS]
            tgt = [t + 2 for r in range(t+1, t + self.tgt_time_len)]
            tgt.insert(0, self.SPECIAL_TOKEN_BOS)
            tgt.append(self.SPECIAL_TOKEN_EOS)
            dataset_inc.append((src, tgt))

        self.dataset_inc = dataset_inc


    def __getitem__(self, item):
        src_ind, tgt_ind = self.dataset_inc[item]
        src = self.get_data(src_ind)
        tgt = self.get_data(tgt_ind, source=False)
        return src, tgt

