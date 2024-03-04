import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import List
from reformer_pytorch import Reformer, ReformerEncDec
from models.HAEA.datasets.denoised_dataset import TimeVocab


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


class Haea(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                num_heads=12, n_encoder_layers=3, n_decoder_layers=3, dropout=0.1, max_var_len=300):
        super().__init__()
        self.in_dim = in_dim

        self.model = nn.Transformer(
            d_model=in_dim,
            nhead=num_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=in_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.embedding = Embedding(max_var_len, in_dim, dropout)
        self.out = LinearDecoder(in_dim, out_dim, dropout=dropout)

    def forward(self, src: torch.Tensor, src_id: torch.Tensor, tgt: torch.Tensor, tgt_id: torch.Tensor, 
                tgt_mask: torch.Tensor, src_var_list: torch.Tensor, tgt_var_list: torch.Tensor):
    
        src_var_seq = self.get_var_seq(src_var_list, src_id, src.device)
        tgt_var_seq = self.get_var_seq(tgt_var_list, tgt_id, tgt.device)

        src = self.embedding(src, src_var_seq) * math.sqrt(self.in_dim)
        tgt = self.embedding(tgt, tgt_var_seq) * math.sqrt(self.in_dim)
    
        transformer_out = self.model(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=None, tgt_key_padding_mask=None)
        out = self.out(transformer_out)
    
        return out


    def get_var_seq(self, var_list: torch.Tensor, indicate: torch.Tensor, device):
        # indicate.shape = (batch, max_len + 2)
        result = []
        var_list = var_list + len(TimeVocab.SPECIAL_TOKENS)

        for batch in indicate:
            seq = []
            for item in batch:
                if item in TimeVocab.SPECIAL_TOKENS:
                    if item == TimeVocab.SPECIAL_TOKEN_MASK:
                        seq.append(torch.full_like(var_list, item, device=device))
                    else:
                        seq.append(torch.tensor([item], device=device))
                else:
                    seq.append(var_list)

            seq = torch.cat(seq, dim=0)
            result.append(seq)
        result = torch.stack(result, dim=0)
        print(result.shape)
        return result


    def get_size(self, tensor:torch.Tensor):
        element_size = tensor.element_size()

        # 텐서 내 요소의 총 개수
        num_elements = tensor.nelement()

        # 총 메모리 사용량 계산 (바이트 단위)
        total_memory_bytes = element_size * num_elements
        print(f'Tensor shape: {tensor.shape}')
        print(f'Element size: {element_size} bytes')
        print(f'Number of elements: {num_elements}')
        print(f'Total memory usage: {total_memory_bytes} bytes')
    
    
    
    

    


