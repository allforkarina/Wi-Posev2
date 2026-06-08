from __future__ import annotations

import torch
from torch import nn


class SymmetricResidualDownsampleBlock(nn.Module):
    """Time-frequency convolution block that downsamples both time and subcarrier axes."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(stride, stride),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=(stride, stride),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main_path(x) + self.shortcut(x))


class WiFlowSpatialEncoder(nn.Module):
    """Spatial CSI encoder from [B, C, 114, 64] to [B, 128, 29, 16]."""

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        if input_channels not in (3, 12):
            raise ValueError("input_channels must be 3 for raw CSI or 12 for the physics feature bank")
        self.input_channels = input_channels
        self.stem_channels = 32

        self.antenna_mixer = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
        )
        self.feature_stem = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=self.stem_channels,
                kernel_size=(3, 5),
                stride=(1, 1),
                padding=(1, 2),
                bias=False,
            ),
            nn.BatchNorm2d(self.stem_channels),
            nn.ReLU(inplace=True),
        )
        self.resblock1 = SymmetricResidualDownsampleBlock(32, 64, stride=2)
        self.resblock2 = SymmetricResidualDownsampleBlock(64, 128, stride=2)
        self.resblock3 = SymmetricResidualDownsampleBlock(128, 128, stride=1)

    def _to_conv_layout(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 2)

    def _to_model_layout(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_conv_layout(x)       # [B, 3, 64, 114]
        x = self.antenna_mixer(x)         # [B, 3, 64, 114]
        x = self.feature_stem(x)          # [B, 32, 64, 114]
        x = self.resblock1(x)             # [B, 64, 32, 57]
        x = self.resblock2(x)             # [B, 128, 16, 29]
        x = self.resblock3(x)             # [B, 128, 16, 29]
        return self._to_model_layout(x)   # [B, 128, 29, 16]
