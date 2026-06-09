from __future__ import annotations

import torch
from torch import nn


CSI_INPUT_CALIBRATION_TYPES = (
    "none",
    "antenna_subcarrier_affine",
    "antenna_subcarrier_dynamic",
)


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


class DynamicAntennaSubcarrierCalibration(nn.Module):
    """Sample-adaptive antenna/subcarrier calibration from CSI amplitude statistics."""

    def __init__(
        self,
        num_antennas: int = 3,
        num_subcarriers: int = 114,
        basis_count: int = 4,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if basis_count <= 0:
            raise ValueError("basis_count must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.basis_count = basis_count
        self.hidden_dim = hidden_dim
        self.controller_input_dim = num_antennas * num_subcarriers * 2

        self.log_scale = nn.Parameter(torch.zeros(1, num_antennas, num_subcarriers, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_antennas, num_subcarriers, 1))
        self.log_scale_basis = nn.Parameter(torch.empty(basis_count, 1, num_antennas, num_subcarriers, 1))
        self.bias_basis = nn.Parameter(torch.empty(basis_count, 1, num_antennas, num_subcarriers, 1))
        nn.init.normal_(self.log_scale_basis, mean=0.0, std=1e-3)
        nn.init.normal_(self.bias_basis, mean=0.0, std=1e-3)

        self.input_projection = nn.Linear(self.controller_input_dim, hidden_dim)
        self.activation = nn.ReLU(inplace=True)
        self.output_projection = nn.Linear(hidden_dim, basis_count)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.latest_basis_weights = torch.empty(0)

    def _controller_input(self, x: torch.Tensor) -> torch.Tensor:
        temporal_mean = x.mean(dim=-1)
        temporal_std = x.std(dim=-1, unbiased=False)
        return torch.cat((temporal_mean.flatten(1), temporal_std.flatten(1)), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.num_antennas or x.shape[2] != self.num_subcarriers:
            raise ValueError(
                "DynamicAntennaSubcarrierCalibration expects input shaped "
                f"[B, {self.num_antennas}, {self.num_subcarriers}, T]"
            )
        controller_input = self._controller_input(x)
        hidden = self.activation(self.input_projection(controller_input))
        basis_weights = self.output_projection(hidden)
        self.latest_basis_weights = basis_weights.detach()

        log_scale_delta = torch.einsum(
            "bk,kaft->baft",
            basis_weights,
            self.log_scale_basis.squeeze(1),
        )
        bias_delta = torch.einsum(
            "bk,kaft->baft",
            basis_weights,
            self.bias_basis.squeeze(1),
        )
        log_scale = self.log_scale + log_scale_delta
        bias = self.bias + bias_delta
        return x * torch.exp(log_scale) + bias


def build_csi_input_calibration(calibration_type: str) -> nn.Module:
    if calibration_type == "none":
        return nn.Identity()
    if calibration_type == "antenna_subcarrier_affine":
        return AntennaSubcarrierAffineCalibration()
    if calibration_type == "antenna_subcarrier_dynamic":
        return DynamicAntennaSubcarrierCalibration()
    raise ValueError(f"calibration_type must be one of {CSI_INPUT_CALIBRATION_TYPES}")
