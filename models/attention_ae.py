import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, var_len, hidden_dim, num_heads=12, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.encoder = nn.Sequential(
            nn.Linear(var_len, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 1),
        )
        
        

    def forward(self, input):
        return self.blocks(input)