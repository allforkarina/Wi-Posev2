from __future__ import annotations

import sys
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import CSI_FEATURE_MODES, WiFlowModel, build_csi_feature_bank  # noqa: E402
from eval import load_checkpoint_model  # noqa: E402


@pytest.fixture()
def temp_dir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="csi_feature_bank_", dir=Path.cwd()))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_raw_csi_feature_mode_preserves_three_channel_input() -> None:
    csi = torch.randn(2, 3, 114, 64)

    features = build_csi_feature_bank(csi, mode="raw")

    assert CSI_FEATURE_MODES == ("raw", "physics_bank")
    assert features.shape == csi.shape
    assert torch.allclose(features, csi)


def test_physics_bank_adds_temporal_antenna_and_subcarrier_views() -> None:
    csi = torch.randn(2, 3, 114, 64)

    features = build_csi_feature_bank(csi, mode="physics_bank")

    raw = features[:, 0:3]
    temporal_residual = features[:, 3:6]
    antenna_difference = features[:, 6:9]
    subcarrier_gradient = features[:, 9:12]
    assert features.shape == (2, 12, 114, 64)
    assert torch.allclose(raw, csi)
    assert torch.allclose(
        temporal_residual.mean(dim=-1),
        torch.zeros_like(temporal_residual.mean(dim=-1)),
        atol=1e-5,
    )
    assert torch.allclose(
        antenna_difference.mean(dim=1),
        torch.zeros_like(antenna_difference.mean(dim=1)),
        atol=1e-5,
    )
    assert torch.allclose(subcarrier_gradient[:, :, 0], torch.zeros_like(subcarrier_gradient[:, :, 0]))


def test_wiflow_model_uses_physics_bank_as_twelve_channel_encoder_input() -> None:
    model = WiFlowModel(csi_feature_mode="physics_bank")
    csi = torch.randn(1, 3, 114, 64)

    with torch.no_grad():
        prediction = model(csi)

    assert model.csi_feature_mode == "physics_bank"
    assert model.spatial_encoder.input_channels == 12
    assert prediction.shape == (1, 18, 2)


def test_eval_rebuilds_physics_bank_model_from_checkpoint_config(temp_dir: Path) -> None:
    checkpoint_path = temp_dir / "physics_bank.pth"
    model = WiFlowModel(csi_feature_mode="physics_bank")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": {
                "axial_mode": "spatial_then_temporal",
                "decoder_type": "joint",
                "csi_feature_mode": "physics_bank",
            },
        },
        checkpoint_path,
    )

    loaded = load_checkpoint_model(checkpoint_path, torch.device("cpu"))

    assert loaded.csi_feature_mode == "physics_bank"
    assert loaded.spatial_encoder.input_channels == 12
