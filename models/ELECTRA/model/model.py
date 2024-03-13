import torch
from torch import nn
from torch.nn import functional as F
import math
from models.ELECTRA.datasets.denoised_dataset import TimeVocab


def get_var_seq(var_list: torch.Tensor, indicate: torch.Tensor, device):
    # indicate.shape = (batch, max_len)
    result = []
    mask_ind = []
    var_len = var_list.size(0)

    for batch in indicate:
        seq = []
        mask = []
        for i, item in enumerate(batch):
            if item == TimeVocab.SPECIAL_TOKEN_MASK:
                    seq.append(torch.full_like(var_list, TimeVocab.SPECIAL_TOKEN_MASK, device=device))
                    mask.extend(range(i*var_len, i*var_len + var_len, 1))
            else:
                seq.append(var_list)

        seq = torch.cat(seq, dim=0)
        result.append(seq)
        mask_ind.append(mask)
            
    result = torch.stack(result, dim=0)
    mask_ind = torch.tensor(mask_ind)
    return result, mask_ind


def positional_encoding(shape, device):
    with torch.no_grad(): 
        batch, time_len, var_len, d_model = shape 
        pe = torch.zeros(batch, time_len, d_model, device=device).float()
        position = torch.arange(0, time_len).float().unsqueeze(1)
        
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

    return pe.repeat_interleave(var_len, dim=1)


class VariableEmbedding(nn.Embedding):
    def __init__(self, var_len, embed_size=768):
        super().__init__(var_len, embed_size)


class LeadTimeEmbedding(nn.Embedding):
    def __init__(self, max_lead_time, embed_size=768):
        super().__init__(max_lead_time, embed_size)


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

    def forward(self, x: torch.Tensor, variable: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        var_emb = self.variable(variable)
        return self.dropout(x + var_emb + pos_emb)


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
    def __init__(self, in_dim: int, num_heads=12, n_layers=3, dropout=0.1):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # src.shape = (batch, time * var_len, hidden)
        x = self.model(x)
        # out.shape = (batch, time * var_len, hidden)
        return x



class Electra(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, 
                 num_heads=12, g_layers=3, d_layers=12, dropout=0.1, max_var_len=300):
        
        self.in_dim = in_dim
        self.generator = CliBERT(in_dim, num_heads, g_layers, dropout)
        self.discriminator = CliBERT(in_dim, num_heads, d_layers, dropout)
        self.embedding = Embedding(max_var_len, in_dim, dropout)
        self.decoder = LinearDecoder(in_dim, out_dim, dropout=dropout)
        self.logits = nn.Linear(in_dim, 1)
    

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, var_list: torch.Tensor, src_id: torch.Tensor):
        # src.shape = (batch, time, var_len, hidden), lead_time.shape = (batch)
        src_pe = positional_encoding(src.shape, src.device)
        src = src.view(src.size(0), -1, src.size(-1))
        tgt = tgt.view(tgt.size(0), -1, tgt.size(-1))
        
        # masked.shape = (batch, mask_size, hidden)
        masked, mask_ind = self.generate(src, src_pe, var_list, src_id)
        for i in range(src.size(0)):
            src[i, mask_ind[i]] = masked[i]

        mlm_loss = torch.sqrt(F.mse_loss(tgt, x))
        x = self.discriminate(x, src_pe, var_list)
        return x, mlm_loss


    def generate(self, x: torch.Tensor, src_pe: torch.Tensor, var_list: torch.Tensor, src_id: torch.Tensor):
        gen_var_seq, mask_ind = get_var_seq(var_list, src_id, x.device)
        gen = self.embedding(x, gen_var_seq, src_pe) * math.sqrt(self.in_dim)
        gen = self.generator(gen)
        gen = self.decoder(gen)
        
        masked = []
        for i in range(gen.size(0)):
            masked.append(gen[i, mask_ind[i]].unsqueeze(0))

        masked = torch.cat(masked, dim=0)
        label = 
        return masked, mask_ind
    

    def discriminate(self, x: torch.Tensor, src_pe: torch.Tensor, var_list: torch.Tensor):
        var_seq = var_list.repeat_interleave(x.size(1), dim=1)
        x = self.embedding(x, var_seq, src_pe) * math.sqrt(self.in_dim)
        x = self.discriminator(x)
        x = self.logits(x)
        return x
    


        

    

    


