import torch
from torch import nn
from torch.nn import functional as F
import math
from models.ELECTRA.datasets.denoised_dataset import TimeVocab


class CliBERT(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, 
                 num_heads=12, n_layers=3, dropout=0.1, max_lead_time=500, max_var_len=300):
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
        self.encoder = nn.TransformerEncoder(
            encoder_layers,
            n_layers
        )
        
        self.embedding = BERTEmbedding(max_var_len, in_dim, dropout)
        self.decoder = LinearDecoder(in_dim, out_dim, dropout=dropout)
    

    def forward(self, x: torch.Tensor, var_list: torch.Tensor, src_id: torch.Tensor):
        # src.shape = (batch, time, var_len, hidden), lead_time.shape = (batch)
        var_seq = get_var_seq(var_list, src_id, x.device)
        src_pe = self.positional_encoding(x.shape, x.device)
        x = x.view(x.size(0), -1, x.size(-1))

        x = self.embedding(x, var_seq, src_pe) * math.sqrt(self.in_dim)
        x = self.encoder(x)
        # out.shape = (batch, var_len, hidden)
        x = self.decoder(x)
        return x


    def positional_encoding(self, shape, device):       
        batch, time_len, var_len, d_model = shape 
        pe = torch.zeros(batch, time_len, d_model, device=device).float()
        pe.require_grad = False
        position = torch.arange(0, time_len).float().unsqueeze(1)
        
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

        return pe.repeat_interleave(var_len, dim=1)


def get_var_seq(var_list: torch.Tensor, indicate: torch.Tensor, device):
    # indicate.shape = (batch, max_len)
    result = []

    for batch in indicate:
        seq = []
        for i, item in enumerate(batch):
            if item == TimeVocab.SPECIAL_TOKEN_MASK:
                    seq.append(torch.full_like(var_list, TimeVocab.SPECIAL_TOKEN_MASK, device=device))
            else:
                seq.append(var_list)

        seq = torch.cat(seq, dim=0)
        
        result.append(seq)
            
    result = torch.stack(result, dim=0)
    return result