import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import List


class VariableEmbedding(nn.Embedding):
    def __init__(self, var_len, embed_size=768):
        super().__init__(var_len, embed_size)


class LeadTimeEmbedding(nn.Embedding):
    def __init__(self, max_lead_time, embed_size=768):
        super().__init__(max_lead_time, embed_size)


class Embedding(nn.Module):
    def __init__(self, max_lead_time, var_len, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.variable = VariableEmbedding(var_len, embed_size)
        self.time = LeadTimeEmbedding(max_lead_time, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, x: torch.Tensor, variable_seq: torch.Tensor, lead_time_seq: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.variable(variable_seq) + self.time(lead_time_seq))


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


class ClimateTransformer(nn.Module):
    def __init__(self, var_list: List[str], in_dim: int, out_dim: int, 
                 num_heads=12, n_layers=3, dropout=0.1, max_lead_time=168, max_var_len=300):
        super().__init__()
        self.var_list = var_list
        self.in_dim = in_dim
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=in_dim*4,
            dropout=dropout,
            batch_first=True,
            activation=F.gelu
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layers,
            n_layers
        )
        
        self.embedding = Embedding(max_lead_time, max_var_len, in_dim, dropout)
        self.decoder = LinearDecoder(in_dim, out_dim, dropout=dropout)
    

    def forward(self, src: torch.Tensor, lead_time: torch.Tensor):
        nn.BCE
        # src.shape = (batch, var_len, hidden), lead_time.shape = (batch)
        lead_time = lead_time.unsqueeze(1).repeat(1, src.size(1))
        var_seq = torch.tensor([self.var_list for _ in range(src.size(0))], device=src.device)
        src = self.embedding(src, var_seq, lead_time) * math.sqrt(self.in_dim)
        out = self.encoder(src)
        # out.shape = (batch, var_len, hidden)
        out = self.decoder(out)
        return out
    

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

    


