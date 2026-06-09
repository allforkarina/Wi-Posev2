from __future__ import annotations

import torch
from torch import nn

SPATIAL_STEM_TYPES = ("baseline", "background_gated")


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


class BackgroundGatedSpatialStem(nn.Module):
    """Separate slow CSI background from temporal residuals before feature mixing."""

    def __init__(
        self,
        input_channels: int,
        stem_channels: int = 32,
        background_kernel_size: int = 9,
    ) -> None:
        super().__init__()
        if background_kernel_size <= 0 or background_kernel_size % 2 == 0:
            raise ValueError("background_kernel_size must be a positive odd integer")
        self.input_channels = input_channels
        self.stem_channels = stem_channels
        self.background_kernel_size = background_kernel_size
        self.latest_diagnostics: dict[str, torch.Tensor] = {}

        self.background_filter = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size=(background_kernel_size, 1),
            padding=(background_kernel_size // 2, 0),
            groups=input_channels,
            bias=False,
        )
        with torch.no_grad():
            self.background_filter.weight.fill_(1.0 / background_kernel_size)

        self.raw_stem = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=stem_channels,
                kernel_size=(3, 5),
                stride=(1, 1),
                padding=(1, 2),
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.residual_stem = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=stem_channels,
                kernel_size=(3, 5),
                stride=(1, 1),
                padding=(1, 2),
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(stem_channels * 2, stem_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        background = self.background_filter(x)
        residual = x - background
        raw_feature = self.raw_stem(x)
        residual_feature = self.residual_stem(residual)
        gate = self.gate(torch.cat((raw_feature, residual_feature), dim=1))
        fused_feature = raw_feature * (1.0 - gate) + residual_feature * gate
        self.latest_diagnostics = {
            "background": background.detach(),
            "residual": residual.detach(),
            "raw_feature": raw_feature.detach(),
            "residual_feature": residual_feature.detach(),
            "gate": gate.detach(),
            "fused_feature": fused_feature.detach(),
        }
        return fused_feature


class WiFlowSpatialEncoder(nn.Module):
    """Spatial CSI encoder from [B, C, 114, 64] to [B, 128, 29, 16]."""

    def __init__(
        self,
        input_channels: int = 3,
        stem_type: str = "baseline",
        background_kernel_size: int = 9,
    ) -> None:
        super().__init__()
        if input_channels not in (3, 12):
            raise ValueError("input_channels must be 3 for raw CSI or 12 for the physics feature bank")
        if stem_type not in SPATIAL_STEM_TYPES:
            raise ValueError(f"stem_type must be one of {SPATIAL_STEM_TYPES}")
        self.input_channels = input_channels
        self.stem_type = stem_type
        self.stem_channels = 32
        self.background_kernel_size = background_kernel_size
        self.latest_diagnostics: dict[str, torch.Tensor] = {}

        if stem_type == "baseline":
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
        else:
            self.background_gated_stem = BackgroundGatedSpatialStem(
                input_channels=input_channels,
                stem_channels=self.stem_channels,
                background_kernel_size=background_kernel_size,
            )
        self.resblock1 = SymmetricResidualDownsampleBlock(32, 64, stride=2)
        self.resblock2 = SymmetricResidualDownsampleBlock(64, 128, stride=2)
        self.resblock3 = SymmetricResidualDownsampleBlock(128, 128, stride=1)

    @property
    def stem(self) -> nn.Module:
        if self.stem_type == "background_gated":
            return self.background_gated_stem
        return self

    def _to_conv_layout(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 2)

    def _to_model_layout(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_conv_layout(x)       # [B, 3, 64, 114]
        if self.stem_type == "baseline":
            x = self.antenna_mixer(x)     # [B, C, 64, 114]
            x = self.feature_stem(x)      # [B, 32, 64, 114]
            self.latest_diagnostics = {"fused_feature": x.detach()}
        else:
            x = self.background_gated_stem(x)
        x = self.resblock1(x)             # [B, 64, 32, 57]
        x = self.resblock2(x)             # [B, 128, 16, 29]
        x = self.resblock3(x)             # [B, 128, 16, 29]
        return self._to_model_layout(x)   # [B, 128, 29, 16]
