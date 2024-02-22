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


class Embedding(nn.Module):
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

    def forward(self, x, variable_seq, position_seq=None):
        if position_seq is not None:
            x = x +  self.variable(variable_seq) + self.position(position_seq)
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


class VariableEncoder(nn.Module):
    def __init__(self, src_var_seq, tgt_var_seq, tgt_time_len, dim_model, out_dim, batch_size, 
                 max_len=4*10, num_heads=12, n_encoder_layers=3, n_decoder_layers=3, dropout=0.1):
        super().__init__()
        self.var_seq = range(var_len)
        self.var_len = var_len
        self.tgt_time_len = tgt_time_len
        self.dim_model = dim_model
        self.batch_size = batch_size

        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=dim_model*2,
            dropout=dropout,
            batch_first=True
        )
        
        self.embedding = Embedding(var_len, max_len, dim_model, dropout)
        self.out = LinearDecoder(dim_model, out_dim, dropout=dropout)
    

    def init_seq(self, device):
        print("init", device)
        self.tgt_mask = self.get_tgt_mask()
        self.tgt_var_seq = self.get_var_seq(self.batch_size, device)
        self.tgt_pos_seq = self.get_pos_seq(self.batch_size, device)


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
    

    def get_var_seq(self, batch_size, device):
        tgt_seq = []

        for _ in range(batch_size):
            t_seq = []

            for _ in range(0, self.tgt_time_len):
                t_seq.extend(self.var_seq)

            tgt_seq.append(t_seq)
        
        tgt_seq = torch.tensor(tgt_seq, device=device)
        return tgt_seq


    def get_pos_seq(self, batch_size, device):
        tgt_pos_seq = []

        for _ in range(batch_size):
            tgt_seq = []

            for i in range(0, self.tgt_time_len):
                tgt_seq.extend([ i for _ in range(self.var_len)])
        
            tgt_pos_seq.append(tgt_seq)
        return torch.tensor(tgt_pos_seq, device=device)
    

    def get_tgt_mask(self) -> torch.tensor:
        matrix = torch.zeros(self.var_len * self.tgt_time_len, self.var_len * self.tgt_time_len)

        for i in range(self.tgt_time_len):
            for j in range(self.var_len):
                inf_idx = min(((i)*self.var_len), self.var_len * self.tgt_time_len)
                matrix[:(i*self.var_len), inf_idx:] = float('-inf')
        return matrix

    
    @torch.no_grad()
    def get_attention_maps(self, x: torch.Tensor):
        """Function for extracting the attention matrices of the whole Transformer for a single batch.

        Input arguments same as the forward pass.
        """
        x = x.view(x.size(0), -1, x.size(3))
        src_var_seq = torch.tensor([self.var_seq for _ in range(self.batch_size)], device=x.device)

        x = self.embedding(x, src_var_seq) * math.sqrt(self.dim_model)

        attention_maps = []
        for layer in self.transformer.encoder.layers:
            _, attn_map = layer.self_attn(query=x, key=x, value=x)
            attention_maps.append(attn_map)
            x = layer(x)

        return attention_maps

    


