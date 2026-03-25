from __future__ import annotations

import torch
from torch import Tensor, nn


class SmallTactileImageEncoder(nn.Module):
    """A lightweight CNN encoder for tactile images.

    Input:  (B, C, H, W)
    Output: (B, embed_dim)

    This is intentionally small vs. using a full vision backbone.
    """

    def __init__(
        self,
        embed_dim: int,
        in_channels: int = 3,
        width: int = 64,
    ) -> None:
        super().__init__()

        def block(cin: int, cout: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(num_groups=8 if cout >= 8 else 1, num_channels=cout),
                nn.SiLU(),
            )

        self.conv = nn.Sequential(
            block(in_channels, width, stride=2),
            block(width, width, stride=1),
            block(width, width * 2, stride=2),
            block(width * 2, width * 2, stride=1),
            block(width * 2, width * 4, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(width * 4, embed_dim, bias=True),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W) tactile image tensor, got {tuple(x.shape)}")
        x = x.to(dtype=torch.float32)
        x = self.conv(x)
        x = self.proj(x)
        return x
