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
    
class SourceEmbedding(nn.Module):
    def __init__(self, var_len, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """ 
        super().__init__()
        self.variable = VariableEmbedding(var_len, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, src, variable_seq):
        x = src + self.variable(variable_seq)
        return self.dropout(x)



class VariableAnalyzer(nn.Module):
    def __init__(self, var_len, time_len, dim_model, predict_dim, batch_size, num_heads=12, n_encoder_layers=3, n_decoder_layers=3, dropout=0.1):
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
        self.tgt_embedding = TargetEmbedding(var_len, time_len, dim_model, dropout)
        self.src_embedding = SourceEmbedding(var_len, dim_model)
        self.out = nn.Linear(dim_model, predict_dim)
        self.tgt_mask = self.get_tgt_mask()

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        # src.shape = (batch, 99, 1450), tgt.shape = (batch, 99 * time_len, 1450)
        src_var_seq, tgt_var_seq = self.get_var_seq(src)
        time_seq = self.get_time_seq(src)

        src = self.src_embedding(src, src_var_seq) * math.sqrt(self.dim_model)
        tgt = self.tgt_embedding(tgt, time_seq, tgt_var_seq) * math.sqrt(self.dim_model)
        tgt_mask = self.tgt_mask.to(src.device)

        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=None, tgt_key_padding_mask=None)
        out = self.out(transformer_out)

        return out
    

    def get_var_seq(self, src: torch.Tensor):
        src_seq = torch.tensor([self.var_seq for _ in range(src.size(0))], device=src.device)
        tgt_seq = []
        for _ in range(src.size(0)):
            seq = []
            for _ in range(0, self.time_len):
                seq.extend(self.var_seq)
            tgt_seq.append(seq)
        
        tgt_seq = torch.tensor(tgt_seq, device=src.device)
        return src_seq, tgt_seq


    def get_time_seq(self, src: torch.Tensor):
        time_seq = []
        for _ in range(src.size(0)):
            seq = []
            for i in range(0, self.time_len):
                seq.extend([ i for _ in range(self.var_len)])
        
            time_seq.append(seq)
        
        return torch.tensor(time_seq, device=src.device)
    

    def get_tgt_mask(self) -> torch.tensor:
        matrix = torch.zeros(self.var_len * self.time_len, self.var_len * self.time_len)
        for i in range(0, self.var_len * self.time_len):
            matrix[i, min(self.var_len*(i+1), self.var_len * self.time_len):] = float('-inf')
        return matrix

    


