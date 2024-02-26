import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import List
from reformer_pytorch import Reformer, ReformerLM, ReformerEncDec

class VariableEmbedding(nn.Embedding):
    def __init__(self, var_len, embed_size=768):
        super().__init__(var_len, embed_size)


class Embedding(nn.Module):
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

    def forward(self, x, variable_seq, position_seq=None):
        if position_seq is not None:
            x = x +  self.variable(variable_seq) + position_seq
            return self.dropout(x)
        else:
            x = x + self.variable(variable_seq)
            return self.dropout(x)


class LinearDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
       return self.seq(x)


class Haea(nn.Module):
    def __init__(self, src_var_list: List[str], tgt_var_list: List[str], time_len: int, 
                in_dim: int, out_dim: int, max_var_len=300, 
                num_heads=12, n_encoder_layers=3, n_decoder_layers=3, dropout=0.1):
        super().__init__()
        self.src_var_list = src_var_list
        self.tgt_var_list = tgt_var_list
        self.time_len = time_len
        self.in_dim = in_dim

        self.encoder = Reformer(
            dim=in_dim,
            depth=n_encoder_layers,
            heads=num_heads,
            causal=False,
            ff_glu=True,
            n_hashes = 4,
            attn_chunks = 8,
            dropout = dropout
        )

        self.decoder = Reformer(
            dim=in_dim,
            depth=n_decoder_layers,
            heads=num_heads,
            causal=False,
            ff_glu=True,
            n_hashes = 4,
            attn_chunks = 8,
            dropout = dropout
        )
        
        self.embedding = Embedding(max_var_len, in_dim, dropout)
        self.out = LinearDecoder(in_dim, out_dim, dropout=dropout)
    

    def init_seq(self, device, batch_size):
        self.tgt_mask = self.get_tgt_mask(device)
        self.src_var_seq, self.tgt_var_seq = self.get_var_seq(batch_size, device)
        self.src_pos_seq = self.positional_encoding(batch_size, self.in_dim, len(self.src_var_seq), self.time_len, device)
        self.tgt_pos_seq = self.positional_encoding(batch_size, self.in_dim, len(self.tgt_var_list), self.time_len, device)


    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        # src.shape = (batch, 1, 99, 1450), tgt.shape = (batch, tgt_time_len, 99, 1450)
        if not hasattr(self, 'src_var_seq'):
            self.init_seq(src.device, src.size(0))
        src, tgt = src.view(src.size(0), -1, src.size(3)), tgt.view(tgt.size(0), -1, tgt.size(3))
        src = self.embedding(src, self.src_var_seq) * math.sqrt(self.in_dim)
        tgt = self.embedding(tgt, self.tgt_var_seq, self.tgt_pos_seq) * math.sqrt(self.in_dim)

        x = self.encoder(src)
        x = self.decoder(tgt, keys=x, input_attn_mask=self.tgt_mask)
        
        out = self.out(x)
        return out


    def get_var_seq(self, batch_size, device):
        src_seq = []
        tgt_seq = []

        for _ in range(batch_size):
            s_seq = []
            t_seq = []

            for _ in range(0, self.time_len):
                s_seq.extend(self.src_var_list)
                t_seq.extend(self.tgt_var_list)

            src_seq.append(s_seq)
            tgt_seq.append(t_seq)
        
        src_seq = torch.tensor(src_seq, device=device)
        tgt_seq = torch.tensor(tgt_seq, device=device)
        return src_seq, tgt_seq


    def positional_encoding(self, batch, d_model, var_len, time_len, device):
        pe = torch.zeros(batch, time_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, time_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

        return pe.repeat_interleave(var_len, dim=1).to(device)
    


    def get_tgt_mask(self, batch, device) -> torch.tensor:
        var_len = len(self.tgt_var_list)
        matrix = torch.zeros(batch, var_len * self.time_len, var_len * self.time_len, device=device)

        for i in range(self.time_len):
            for _ in range(var_len):
                inf_idx = min(((i)*var_len), var_len * self.time_len)
                matrix[:, :(i*var_len), inf_idx:] = float('-inf')
        return matrix
    
    

    


