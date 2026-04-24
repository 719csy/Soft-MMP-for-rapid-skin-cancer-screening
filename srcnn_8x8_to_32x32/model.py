from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN8x8To32x32(nn.Module):
    """Classic three-layer SRCNN after bicubic upsampling to the target grid."""

    def __init__(self, input_size: int = 8, target_size: int = 32, channels: tuple[int, int] = (64, 32)):
        super().__init__()
        self.input_size = int(input_size)
        self.target_size = int(target_size)
        hidden1, hidden2 = channels
        self.conv1 = nn.Conv2d(1, hidden1, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(hidden1, hidden2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(hidden2, 1, kernel_size=5, padding=2)

    def upsample(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == (self.target_size, self.target_size):
            return x
        return F.interpolate(
            x,
            size=(self.target_size, self.target_size),
            mode="bicubic",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.sigmoid(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return torch.sigmoid(self.conv3(x))

