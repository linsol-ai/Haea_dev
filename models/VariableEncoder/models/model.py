import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import List

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

    def forward(self, x, variable_seq):
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


class VariableEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, max_var_len=300, num_heads=12, n_encoder_layers=3, n_decoder_layers=3, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim

        self.transformer = nn.Transformer(
            d_model=in_dim,
            nhead=num_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=in_dim*2,
            dropout=dropout,
            batch_first=True
        )
        
        self.embedding = Embedding(max_var_len, in_dim, dropout)
        self.out = LinearDecoder(in_dim, out_dim, dropout=dropout)


    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_var_seq: torch.Tensor, tgt_var_seq: torch.Tensor, tgt_mask: torch.Tensor):
        # src.shape = (batch, 1, 99, 1450), tgt.shape = (batch, tgt_time_len, 99, 1450)
        pe = self.positional_encoding(tgt.size(0))
        src, tgt = src.squeeze(1), tgt.view(tgt.size(0), -1, tgt.size(3))
        src = self.embedding(src, src_var_seq) * math.sqrt(self.in_dim)
        tgt = tgt + pe
        tgt = self.embedding(tgt, tgt_var_seq) * math.sqrt(self.in_dim)

        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=None, tgt_key_padding_mask=None)
        out = self.out(transformer_out)
        return out


    def positional_encoding(self, batch, time_len, var_len, d_model):
        pe = torch.zeros(batch, time_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, time_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

        return pe.repeat_interleave(var_len, dim=1)


    @torch.no_grad()
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Function for extracting the attention matrices of the whole Transformer for a single batch.

        Input arguments same as the forward pass.
        """
        x = x.view(x.size(0), -1, x.size(3))
        src_seq = torch.tensor([self.src_var_list for _ in range(x.size(0))], device=x.device)

        x = self.embedding(x, src_seq) * math.sqrt(self.in_dim)

        attention_maps = []
        for layer in self.transformer.encoder.layers:
            _, attn_map = layer.self_attn(query=x, key=x, value=x)
            attention_maps.append(attn_map)
            x = layer(x)

        return torch.stack(attention_maps, dim=0)

    


