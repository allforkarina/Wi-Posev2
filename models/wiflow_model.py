from __future__ import annotations

import torch
from torch import nn

from .wiflow_axial_encoder import WiFlowAxialEncoder
from .wiflow_heatmap_decoder import WiFlowMSFNDecoder
from .wiflow_hierarchical_joint_decoder import WiFlowHierarchicalJointDecoder
from .wiflow_joint_decoder import WiFlowJointDecoder
from .wiflow_spatial_encoder import WiFlowSpatialEncoder
from pose_targets import decode_pcm_argmax

DECODER_TYPES = ("joint", "hierarchical", "heatmap_msfn")


class WiFlowModel(nn.Module):
    """End-to-end WiFlow model that maps CSI features to OpenPose18 coordinates."""

    def __init__(
        self,
        input_channels: int = 3,
        axial_mode: str = "spatial_then_temporal",
        decoder_type: str = "joint",
        heatmap_size: int = 36,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if decoder_type not in DECODER_TYPES:
            raise ValueError(f"decoder_type must be one of {DECODER_TYPES}")
        self.input_channels = input_channels
        self.axial_mode = axial_mode
        self.decoder_type = decoder_type
        self.heatmap_size = heatmap_size
        self.spatial_encoder = WiFlowSpatialEncoder(input_channels=input_channels)
        self.axial_encoder = WiFlowAxialEncoder(mode=axial_mode, dropout=dropout)
        if decoder_type == "joint":
            self.decoder = WiFlowJointDecoder(dropout=dropout)
        elif decoder_type == "hierarchical":
            self.decoder = WiFlowHierarchicalJointDecoder()
        else:
            self.decoder = WiFlowMSFNDecoder(heatmap_size=heatmap_size)

    def decode_features(self, x: torch.Tensor):
        decoder_output = self.decoder(x)
        if self.decoder_type != "heatmap_msfn":
            return decoder_output
        stages = decoder_output
        keypoints = decode_pcm_argmax(stages[-1]["pcm"])
        return {"keypoints": keypoints, "stages": stages}

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError("WiFlowModel expects input shaped [B, 3, 114, 64]")
        x = self.spatial_encoder(x)
        x = self.axial_encoder(x)
        return self.decode_features(x)