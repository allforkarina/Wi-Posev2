from __future__ import annotations

import torch
from torch import nn


CSI_INPUT_CALIBRATION_TYPES = ("none", "antenna_subcarrier_affine")


class AntennaSubcarrierAffineCalibration(nn.Module):
    """Per-antenna/subcarrier affine calibration for CSI amplitude tensors."""

    def __init__(self, num_antennas: int = 3, num_subcarriers: int = 114) -> None:
        super().__init__()
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.log_scale = nn.Parameter(torch.zeros(1, num_antennas, num_subcarriers, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_antennas, num_subcarriers, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.num_antennas or x.shape[2] != self.num_subcarriers:
            raise ValueError(
                "AntennaSubcarrierAffineCalibration expects input shaped "
                f"[B, {self.num_antennas}, {self.num_subcarriers}, T]"
            )
        return x * torch.exp(self.log_scale) + self.bias


def build_csi_input_calibration(calibration_type: str) -> nn.Module:
    if calibration_type == "none":
        return nn.Identity()
    if calibration_type == "antenna_subcarrier_affine":
        return AntennaSubcarrierAffineCalibration()
    raise ValueError(f"calibration_type must be one of {CSI_INPUT_CALIBRATION_TYPES}")
