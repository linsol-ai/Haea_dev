import torch
from torch import nn
from torch.nn import functional as F
import math


class VariableEmbedding(nn.Embedding):
    def __init__(self, var_len, embed_size=768):
        super().__init__(var_len, embed_size)


class PositionalEmbedding(nn.Embedding):
    def __init__(self, time_len, embed_size=768):
        super().__init__(time_len, embed_size)


class TargetEmbedding(nn.Module):
    def __init__(self, var_len, time_len, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.variable = VariableEmbedding(var_len, embed_size)
        self.position = PositionalEmbedding(time_len, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, tgt, position_seq, variable_seq):
        x = tgt + self.position(position_seq) + self.variable(variable_seq)
        return self.dropout(x)
    
class TargetEmbedding(nn.Module):
    def __init__(self, var_len, time_len, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.variable = VariableEmbedding(var_len, embed_size)
        self.position = PositionalEmbedding(time_len, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, tgt, position_seq, variable_seq):
        x = tgt + self.position(position_seq) + self.variable(variable_seq)
        return self.dropout(x)



class VariableAnalyzer(nn.Module):
    def __init__(self, var_len, time_len, dim_model, num_heads=12, n_encoder_layers=3, n_decoder_layers=3, dropout=0.1):
        super().__init__()
        self.var_seq = range(var_len)
        self.var_len = var_len
        self.time_len = time_len
        self.dim_model = dim_model
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=dim_model*4,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True
        )
        self.tgt_embedding = Embedding(var_len, time_len, dim_model, dropout)
        self.src_embedding = VariableEmbedding(var_len, dim_model)
        self.out = nn.Linear(dim_model, dim_model)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        var_seq = torch.tensor([var_seq for _ in range(src.size(0))])
        time_seq = self.get_time_seq(src)

        src = self.src_embedding(var_seq) * math.sqrt(self.dim_model)
        tgt = self.tgt_embedding(time_seq, var_seq) * math.sqrt(self.dim_model)
        tgt_mask = self.get_tgt_mask(src)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=None, tgt_key_padding_mask=None)
        out = self.out(transformer_out)
        return out


    def get_time_seq(self, src: torch.Tensor):
        time_seq = []
        for i in range(self.time_len):
            seq = [i for _ in range(src.size(1))]
            time_seq.extend(seq)
        return torch.tensor([time_seq for _ in range(src.size(0))])
    
    def get_tgt_mask(self, src:torch.Tensor) -> torch.Tensor:
        mask = []
        for k in range(self.time_len):
            time_mask = []
            for i in range(self.time_len):
                if k == i :
                    seq = [i for _ in range(src.size(1))]
                else:
                    seq = [0 for _ in range(src.size(1))]
                time_mask.extend(seq)
            mask.append(time_mask)

        mask = torch.tensor([mask for _ in range(src.size(0))])
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        return mask

    


