import torch
import torch.nn as nn

class ResidualBlock(torch.nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x
