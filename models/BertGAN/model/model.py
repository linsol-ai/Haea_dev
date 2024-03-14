import torch
from torch import nn
from torch.nn import functional as F
import math

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

    def forward(self, x) -> torch.Tensor:
       # x.shape = (batch, time * var, hidden)
       return self.seq(x)

class CliBERT(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads=12, n_layers=3, dropout=0.1, 
                 max_lead_time=500, max_var_len=300):
        super().__init__()

        self.in_dim = in_dim
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=in_dim*4,
            dropout=dropout,
            batch_first=True,
            activation=F.gelu
        )
        self.model = nn.TransformerEncoder(
            encoder_layers,
            n_layers
        )
        self.decoder = LinearDecoder(in_dim, out_dim, dropout=dropout)
    

    def forward(self, x: torch.Tensor, lead_time: torch.Tensor, var_list: torch.Tensor):
        # src.shape = (batch, time, var_len, hidden), lead_time.shape = (batch)
        var_seq = var_list.repeat_interleave(x.size(1), dim=0).unsqueeze(0).repeat_interleave(x.size(0), dim=0)
        src_pe = positional_encoding(x.shape, x.device)
        x = x.view(x.size(0), -1, x.size(-1))
        lead_time = lead_time.unsqueeze(1).repeat(1, x.size(1))

        x = self.embedding(x, var_seq, lead_time, src_pe) * math.sqrt(self.in_dim)
        x = self.model(x)
        # out.shape = (batch, var_len, hidden)
        x = self.decoder(x)
        return x