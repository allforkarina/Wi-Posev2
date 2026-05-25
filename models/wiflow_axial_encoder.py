from __future__ import annotations

import torch
from torch import nn


AXIAL_ENCODER_MODES: tuple[str, ...] = (
    "spatial_then_temporal",
    "temporal_then_spatial",
    "parallel_sum",
    "parallel_concat",
)


class WiFlowAxialEncoder(nn.Module):
    """Axial attention encoder from [B, 128, 29, 10] to [B, 256, 29, 10]."""

    def __init__(self, mode: str = "spatial_then_temporal", dropout: float = 0.1) -> None:
        super().__init__()
        if mode not in AXIAL_ENCODER_MODES:
            raise ValueError(f"Unsupported axial encoder mode {mode!r}; choose from {AXIAL_ENCODER_MODES}")
        self.mode = mode
        self.input_channels = 128
        self.output_channels = 256
        self.num_heads = 8
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=self.input_channels,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.spatial_norm = nn.LayerNorm(self.input_channels)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.input_channels,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.temporal_norm = nn.LayerNorm(self.input_channels)
        self.channel_projection = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=1)
        if self.mode == "parallel_concat":
            self.concat_projection = nn.Conv2d(self.input_channels * 2, self.output_channels, kernel_size=1)

    def _prepare_spatial_attention_input(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, spatial_tokens, temporal = x.shape
        return x.permute(0, 3, 2, 1).reshape(batch_size * temporal, spatial_tokens, channels)

    def _restore_spatial_attention_output(
        self,
        x: torch.Tensor,
        batch_size: int,
        spatial_tokens: int,
        temporal: int,
    ) -> torch.Tensor:
        return x.reshape(batch_size, temporal, spatial_tokens, self.input_channels).permute(0, 3, 2, 1)

    def _prepare_temporal_attention_input(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, spatial_tokens, temporal = x.shape
        return x.permute(0, 2, 3, 1).reshape(batch_size * spatial_tokens, temporal, channels)

    def _restore_temporal_attention_output(
        self,
        x: torch.Tensor,
        batch_size: int,
        spatial_tokens: int,
        temporal: int,
    ) -> torch.Tensor:
        return x.reshape(batch_size, spatial_tokens, temporal, self.input_channels).permute(0, 3, 1, 2)

    def _apply_spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, spatial_tokens, temporal = x.shape

        spatial_input = self._prepare_spatial_attention_input(x)
        spatial_output, _ = self.spatial_attention(
            spatial_input,
            spatial_input,
            spatial_input,
            need_weights=True,
        )
        spatial_output = self.spatial_norm(spatial_output + spatial_input)
        return self._restore_spatial_attention_output(spatial_output, batch_size, spatial_tokens, temporal)

    def _apply_temporal_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, spatial_tokens, temporal = x.shape

        temporal_input = self._prepare_temporal_attention_input(x)
        temporal_output, _ = self.temporal_attention(
            temporal_input,
            temporal_input,
            temporal_input,
            need_weights=True,
        )
        temporal_output = self.temporal_norm(temporal_output + temporal_input)
        return self._restore_temporal_attention_output(temporal_output, batch_size, spatial_tokens, temporal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "spatial_then_temporal":
            x = self._apply_spatial_attention(x)
            x = self._apply_temporal_attention(x)
            return self.channel_projection(x)
        if self.mode == "temporal_then_spatial":
            x = self._apply_temporal_attention(x)
            x = self._apply_spatial_attention(x)
            return self.channel_projection(x)

        spatial_output = self._apply_spatial_attention(x)
        temporal_output = self._apply_temporal_attention(x)
        if self.mode == "parallel_sum":
            return self.channel_projection(spatial_output + temporal_output)
        return self.concat_projection(torch.cat((spatial_output, temporal_output), dim=1))
