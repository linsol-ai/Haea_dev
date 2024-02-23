import torch
from torch import nn
from torch.nn import functional as F
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class LinearEncoder(nn.Module):
    def __init__(self, in_dim, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.LayerNorm(in_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 4, 1),
        )

    def forward(self, x):
       # x.shape = (batch, )
       return self.seq(x)
    

class LinearDecoder(nn.Module):
    def __init__(self, out_dim, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1, out_dim // 4),
            nn.LayerNorm(out_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 4, out_dim // 2),
            nn.LayerNorm(out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, out_dim),
        )

    def forward(self, x):
       return self.seq(x)


class Haea(nn.Module):
    def __init__(self, var_len, dim_model, batch_size, max_len=24*30, num_heads=12, n_encoder_layers=3, n_decoder_layers=3, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.dim_model = dim_model
        self.batch_size = batch_size

        self.positional_encoder = PositionalEmbedding(dim_model=dim_model, dropout_p=dropout, max_len=max_len)

        self.bart = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=dim_model*4,
            activation=F.gelu,
            dropout=dropout,
            batch_first=True
        )
        
        self.encoder = LinearEncoder(var_len, dropout=dropout)
        self.decoder = LinearDecoder(var_len, dropout=dropout)
    

    def init_seq(self, device):
        print("init", device)
        self.tgt_mask = self.get_tgt_mask()


    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        # src.shape = (batch, 1, 99, 1450), tgt.shape = (batch, tgt_time_len, 99, 1450)
        src, tgt = src.squeeze(1), tgt.view(tgt.size(0), -1, tgt.size(3))
        src_var_seq = torch.tensor([self.var_seq for _ in range(self.batch_size)], device=src.device)

        src = self.embedding(src, src_var_seq) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt, self.tgt_var_seq, self.tgt_pos_seq) * math.sqrt(self.dim_model)
        tgt_mask = self.tgt_mask.to(src.device)

        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=None, tgt_key_padding_mask=None)
        out = self.out(transformer_out)
        return out
    

    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        return mask


    


