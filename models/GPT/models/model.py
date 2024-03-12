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
    

class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNetwork, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.normal_(self.linear2.weight, std=0.02)

    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len, d_model)

        outputs = self.gelu(self.linear1(inputs))
        # |outputs| : (batch_size, seq_len, d_ff)
        outputs = self.linear2(outputs)
        # |outputs| : (batch_size, seq_len, d_model)

        return outputs
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attn_pdrop, resid_pdrop):
        super(DecoderLayer, self).__init__()

        self.mha = nn.MultiheadAttention(d_model, n_heads, attn_pdrop)
        self.dropout1 = nn.Dropout(resid_pdrop)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-5)    

        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout2 = nn.Dropout(resid_pdrop)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-5)    

    def forward(self, inputs, attn_mask):
        # |inputs| : (batch_size, seq_len, d_model)
        # |attn_mask| : (batch_size, seq_len, seq_len)
        
        attn_outputs, attn_weights = self.mha(inputs, inputs, inputs, attn_mask)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(inputs + attn_outputs)
        # |attn_outputs| : (batch_size, seq_len, d_model)
        # |attn_weights| : (batch_size, n_heads, q_len(=seq_len), k_len(=seq_len))
        
        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)
        # |ffn_outputs| : (batch_size, seq_len, d_model)
        
        return ffn_outputs, attn_weights


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, n_layers, n_heads, d_ff, embd_pdrop, attn_pdrop, resid_pdrop, pad_id):
        super(TransformerDecoder, self).__init__()
        self.pad_id = pad_id

        # layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(embd_pdrop)
        self.pos_embedding = nn.Embedding(seq_len+1, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, attn_pdrop, resid_pdrop) for _ in range(n_layers)])
        
        nn.init.normal_(self.embedding.weight, std=0.02)
        
    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len)
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1) + 1
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)
        # |positions| : (batch_size, seq_len)

        outputs = self.dropout(self.embedding(inputs)) + self.pos_embedding(positions)
        # |outputs| : (batch_size, seq_len, d_model)
        
        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)
        # |attn_pad_mask| : (batch_size, seq_len, seq_len)
        subsequent_mask = self.get_attention_subsequent_mask(inputs).to(device=attn_pad_mask.device)
        # |subsequent_mask| : (batch_size, seq_len, seq_len)
        attn_mask = torch.gt((attn_pad_mask.to(dtype=subsequent_mask.dtype) + subsequent_mask), 0)
        # |attn_mask| : (batch_size, seq_len, seq_len)
        
        attention_weights = []
        for layer in self.layers:
            outputs, attn_weights = layer(outputs, attn_mask)
            # |outputs| : (batch_size, seq_len, d_model)
            # |attn_weights| : (batch_size, n_heads, seq_len, seq_len)
            attention_weights.append(attn_weights)
        
        return outputs, attention_weights    
        
    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)

        return attn_pad_mask
    
    def get_attention_subsequent_mask(self, q):
        bs, q_len = q.size()
        subsequent_mask = torch.ones(bs, q_len, q_len).triu(diagonal=1)
        # |subsequent_mask| : (batch_size, q_len, q_len)
        
        return subsequent_mask
    
    


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
        src_pe = self.positional_encoding(src.shape, src.device)
        tgt_pe = self.positional_encoding(tgt.shape, tgt.device)

        src, tgt = src.view(src.size(0), -1, src.size(-1)), tgt.view(tgt.size(0), -1, tgt.size(-1))
        src = (self.embedding(src, src_var_seq) + src_pe) * math.sqrt(self.in_dim)
        tgt =  (self.embedding(tgt, tgt_var_seq) + tgt_pe) * math.sqrt(self.in_dim)

        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=None, tgt_key_padding_mask=None)
        out = self.out(transformer_out)
        return out


    def positional_encoding(self, shape, device):       
        batch, time_len, var_len, d_model = shape 
        pe = torch.zeros(batch, time_len, d_model, device=device).float()
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

    


