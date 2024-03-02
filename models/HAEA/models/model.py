import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import List
from reformer_pytorch import Reformer, ReformerEncDec
from models.HAEA.datasets.denoised_dataset import TimeVocab


class VariableEmbedding(nn.Embedding):
    def __init__(self, var_len, embed_size=768):
        super().__init__(var_len, embed_size, padding_idx=TimeVocab.SPECIAL_TOKEN_PAD)


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
    def __init__(self, vocab: TimeVocab, in_dim: int, out_dim: int, bucket_size=64,
                num_heads=12, n_encoder_layers=3, n_decoder_layers=3, dropout=0.1, max_var_len=300, max_batch_size=64):
        super().__init__()
        self.vocab = vocab
        self.in_dim = in_dim

        self.encoder = Reformer(
            dim=in_dim,
            bucket_size=bucket_size,
            depth=n_encoder_layers,
            heads=num_heads,
            causal=False,
            ff_glu=True,
            n_hashes = 4,
            attn_chunks = 8,
            ff_dropout=dropout
        )

        self.decoder = Reformer(
            dim=in_dim,
            bucket_size=bucket_size,
            depth=n_decoder_layers,
            heads=num_heads,
            causal=False,
            ff_glu=True,
            n_hashes = 4,
            attn_chunks = 8,
            ff_dropout=dropout
        )
        
        self.embedding = Embedding(max_var_len, in_dim, dropout)
        self.out = LinearDecoder(in_dim, out_dim, dropout=dropout)
        self.tgt_mask = self.vocab.tgt_mask.unsqueeze(dim=0).expand(max_batch_size, -1, -1).to()
        self.mask = self.vocab.mask.unsqueeze(dim=0).expand(max_batch_size, -1, -1)


    def forward(self, src: torch.Tensor, src_id: torch.Tensor, tgt: torch.Tensor, tgt_id: torch.Tensor):
        # src.shape = (batch, 1, 99, 1450), tgt.shape = (batch, tgt_time_len, 99, 1450)
        src_var_seq = self.get_var_seq(self.vocab.src_var_list, src_id, self.vocab.src_pad, src.device)
        tgt_var_seq = self.get_var_seq(self.vocab.tgt_var_list, tgt_id, self.vocab.tgt_pad, tgt.device)

        src = self.embedding(src, src_var_seq) * math.sqrt(self.in_dim)
        tgt = self.embedding(tgt, tgt_var_seq) * math.sqrt(self.in_dim)
        mask = self.mask[:src.size(0)]

        x = self.encoder(src, input_mask=mask)
        x = self.decoder(tgt, keys=x, context_mask=mask, input_attn_mask=self.tgt_mask[:src.size(0)])
        out = self.out(x)
        return out


    def get_var_seq(self, var_list: torch.Tensor, indicate: torch.Tensor, pad_len, device):
        # indicate.shape = (batch, max_len + 2)
        result = []
        var_list = var_list + len(TimeVocab.SPECIAL_TOKENS)

        for batch in indicate:
            seq = []
            for item in batch:
                if item in TimeVocab.SPECIAL_TOKENS:
                    if item == TimeVocab.SPECIAL_TOKEN_MASK:
                        seq.extend([item for _ in range(var_list.size(0))])
                    else:
                        seq.append(item)
                else:
                    seq.extend(var_list)
            seq.extend([TimeVocab.SPECIAL_TOKEN_PAD for _ in range(pad_len)])
            result.append(seq)
        result = torch.tensor(result, device=device)
        return result
    
    
    
    

    


