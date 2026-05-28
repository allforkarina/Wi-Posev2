from __future__ import annotations

import torch
from torch import nn

from .wiflow_axial_encoder import WiFlowAxialEncoder
from .wiflow_hierarchical_joint_decoder import WiFlowHierarchicalJointDecoder
from .wiflow_joint_decoder import WiFlowJointDecoder
from .wiflow_spatial_encoder import WiFlowSpatialEncoder

DECODER_TYPES = ("joint", "hierarchical")


class WiFlowModel(nn.Module):
    """End-to-end WiFlow model that maps CSI features to OpenPose18 coordinates."""

    def __init__(
        self,
        input_channels: int = 3,
        axial_mode: str = "spatial_then_temporal",
        decoder_type: str = "joint",
    ) -> None:
        super().__init__()
        if decoder_type not in DECODER_TYPES:
            raise ValueError(f"decoder_type must be one of {DECODER_TYPES}")
        self.input_channels = input_channels
        self.axial_mode = axial_mode
        self.decoder_type = decoder_type
        self.spatial_encoder = WiFlowSpatialEncoder(input_channels=input_channels)
        self.axial_encoder = WiFlowAxialEncoder(mode=axial_mode)
        if decoder_type == "joint":
            self.decoder = WiFlowJointDecoder()
        elif decoder_type == "hierarchical":
            self.decoder = WiFlowHierarchicalJointDecoder()

    def decode_features(self, x: torch.Tensor):
        return self.decoder(x)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError("WiFlowModel expects input shaped [B, 3, 114, 64]")
        x = self.spatial_encoder(x)
        x = self.axial_encoder(x)
        return self.decode_features(x)