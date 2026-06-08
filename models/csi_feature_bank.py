from __future__ import annotations

import torch


CSI_FEATURE_MODES = ("raw", "physics_bank")
CSI_FEATURE_CHANNELS = {
    "raw": 3,
    "physics_bank": 12,
}


def csi_feature_input_channels(mode: str) -> int:
    if mode not in CSI_FEATURE_CHANNELS:
        raise ValueError(f"mode must be one of {CSI_FEATURE_MODES}, got {mode!r}")
    return CSI_FEATURE_CHANNELS[mode]


def _match_reference_scale(
    view: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    view_std = view.flatten(1).std(dim=1, keepdim=True).view(-1, 1, 1, 1).clamp_min(eps)
    reference_std = reference.flatten(1).std(dim=1, keepdim=True).view(-1, 1, 1, 1).clamp_min(eps)
    return view / view_std * reference_std


def build_csi_feature_bank(x: torch.Tensor, mode: str = "raw") -> torch.Tensor:
    if mode not in CSI_FEATURE_MODES:
        raise ValueError(f"mode must be one of {CSI_FEATURE_MODES}, got {mode!r}")
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("CSI feature bank expects raw input shaped [B, 3, 114, 64]")
    if mode == "raw":
        return x

    temporal_residual = x - x.mean(dim=-1, keepdim=True)
    antenna_difference = x - x.mean(dim=1, keepdim=True)
    subcarrier_gradient = torch.zeros_like(x)
    subcarrier_gradient[:, :, 1:, :] = x[:, :, 1:, :] - x[:, :, :-1, :]

    return torch.cat(
        (
            x,
            _match_reference_scale(temporal_residual, x),
            _match_reference_scale(antenna_difference, x),
            _match_reference_scale(subcarrier_gradient, x),
        ),
        dim=1,
    )
