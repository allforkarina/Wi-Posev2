from __future__ import annotations

import torch
from torch import nn

from .csi_feature_bank import CSI_FEATURE_MODES, build_csi_feature_bank, csi_feature_input_channels
from .csi_input_calibration import CSI_INPUT_CALIBRATION_TYPES, build_csi_input_calibration
from .wiflow_axial_encoder import WiFlowAxialEncoder
from .wiflow_hierarchical_joint_decoder import WiFlowHierarchicalJointDecoder
from .wiflow_joint_decoder import WiFlowJointDecoder
from .wiflow_spatial_encoder import WiFlowSpatialEncoder
from .wrist_refiner import WristRefinementHead

DECODER_TYPES = ("joint", "hierarchical")


class WiFlowModel(nn.Module):
    """End-to-end WiFlow model that maps CSI features to OpenPose18 coordinates."""

    def __init__(
        self,
        input_channels: int = 3,
        axial_mode: str = "spatial_then_temporal",
        decoder_type: str = "joint",
        csi_feature_mode: str = "raw",
        spatial_stem_type: str = "baseline",
        background_kernel_size: int = 9,
        input_calibration: str = "none",
        wrist_refinement: bool = False,
    ) -> None:
        super().__init__()
        if decoder_type not in DECODER_TYPES:
            raise ValueError(f"decoder_type must be one of {DECODER_TYPES}")
        if input_channels != 3:
            raise ValueError("WiFlowModel expects three raw CSI amplitude channels before feature-bank expansion")
        if csi_feature_mode not in CSI_FEATURE_MODES:
            raise ValueError(f"csi_feature_mode must be one of {CSI_FEATURE_MODES}")
        if input_calibration not in CSI_INPUT_CALIBRATION_TYPES:
            raise ValueError(f"input_calibration must be one of {CSI_INPUT_CALIBRATION_TYPES}")
        self.input_channels = input_channels
        self.axial_mode = axial_mode
        self.decoder_type = decoder_type
        self.csi_feature_mode = csi_feature_mode
        self.spatial_stem_type = spatial_stem_type
        self.background_kernel_size = background_kernel_size
        self.input_calibration_type = input_calibration
        self.wrist_refinement = wrist_refinement
        self.input_calibration = build_csi_input_calibration(input_calibration)
        self.encoder_input_channels = csi_feature_input_channels(csi_feature_mode)
        self.spatial_encoder = WiFlowSpatialEncoder(
            input_channels=self.encoder_input_channels,
            stem_type=spatial_stem_type,
            background_kernel_size=background_kernel_size,
        )
        self.axial_encoder = WiFlowAxialEncoder(mode=axial_mode)
        if decoder_type == "joint":
            self.decoder = WiFlowJointDecoder()
        elif decoder_type == "hierarchical":
            self.decoder = WiFlowHierarchicalJointDecoder()
        if wrist_refinement:
            self.wrist_refiner = WristRefinementHead(embedding_dim=self.decoder.embedding_dim)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("WiFlowModel expects input shaped [B, 3, 114, 64]")
        x = self.input_calibration(x)
        x = build_csi_feature_bank(x, mode=self.csi_feature_mode)
        x = self.spatial_encoder(x)
        return self.axial_encoder(x)

    def decode_features(
        self,
        x: torch.Tensor,
        return_decoder_features: bool = False,
    ):
        if not self.wrist_refinement:
            return self.decoder(x, return_features=return_decoder_features)

        coordinates, decoder_features = self.decoder(x, return_features=True)
        coordinates = self.wrist_refiner(coordinates, decoder_features)
        if return_decoder_features:
            return coordinates, decoder_features
        return coordinates

    def forward(
        self,
        x: torch.Tensor,
        return_decoder_features: bool = False,
    ):
        x = self.encode_features(x)
        return self.decode_features(x, return_decoder_features=return_decoder_features)
