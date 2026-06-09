from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import load_checkpoint_model  # noqa: E402
from models import SPATIAL_STEM_TYPES, WiFlowModel, WiFlowSpatialEncoder  # noqa: E402
from scripts.stem_feature_diagnostic import summarize_stem_diagnostics  # noqa: E402


@pytest.fixture()
def temp_dir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="background_gated_stem_", dir=Path.cwd()))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_baseline_spatial_stem_keeps_legacy_encoder_shape() -> None:
    encoder = WiFlowSpatialEncoder(input_channels=3)
    csi = torch.randn(2, 3, 114, 64)

    with torch.no_grad():
        features = encoder(csi)

    assert SPATIAL_STEM_TYPES == ("baseline", "background_gated")
    assert encoder.stem_type == "baseline"
    assert features.shape == (2, 128, 29, 16)


def test_background_gated_spatial_stem_exposes_physical_branches() -> None:
    encoder = WiFlowSpatialEncoder(
        input_channels=3,
        stem_type="background_gated",
        background_kernel_size=9,
    )
    csi = torch.randn(2, 3, 114, 64)

    with torch.no_grad():
        features = encoder(csi)

    diagnostics = encoder.stem.latest_diagnostics
    assert encoder.stem_type == "background_gated"
    assert encoder.background_kernel_size == 9
    assert features.shape == (2, 128, 29, 16)
    assert diagnostics["background"].shape == (2, 3, 64, 114)
    assert diagnostics["residual"].shape == (2, 3, 64, 114)
    assert diagnostics["raw_feature"].shape == (2, 32, 64, 114)
    assert diagnostics["residual_feature"].shape == (2, 32, 64, 114)
    assert diagnostics["gate"].shape == (2, 32, 64, 114)
    assert torch.all(diagnostics["gate"] >= 0.0)
    assert torch.all(diagnostics["gate"] <= 1.0)


def test_wiflow_model_forwards_background_gated_stem_config() -> None:
    model = WiFlowModel(spatial_stem_type="background_gated", background_kernel_size=7)
    csi = torch.randn(1, 3, 114, 64)

    with torch.no_grad():
        prediction = model(csi)

    assert model.spatial_stem_type == "background_gated"
    assert model.background_kernel_size == 7
    assert model.spatial_encoder.stem_type == "background_gated"
    assert prediction.shape == (1, 18, 2)


def test_eval_rebuilds_background_gated_stem_from_checkpoint_config(temp_dir: Path) -> None:
    checkpoint_path = temp_dir / "background_gated.pth"
    model = WiFlowModel(spatial_stem_type="background_gated", background_kernel_size=7)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": {
                "axial_mode": "spatial_then_temporal",
                "decoder_type": "joint",
                "csi_feature_mode": "raw",
                "spatial_stem_type": "background_gated",
                "background_kernel_size": 7,
            },
        },
        checkpoint_path,
    )

    loaded = load_checkpoint_model(checkpoint_path, torch.device("cpu"))

    assert loaded.spatial_stem_type == "background_gated"
    assert loaded.background_kernel_size == 7
    assert loaded.spatial_encoder.stem_type == "background_gated"


def test_stem_diagnostic_summary_reports_gate_and_branch_correlations() -> None:
    diagnostics = {
        "background": torch.randn(3, 3, 64, 114),
        "residual": torch.randn(3, 3, 64, 114),
        "raw_feature": torch.randn(3, 32, 64, 114),
        "residual_feature": torch.randn(3, 32, 64, 114),
        "fused_feature": torch.randn(3, 32, 64, 114),
        "gate": torch.full((3, 32, 64, 114), 0.25),
    }
    keypoints = torch.randn(3, 18, 2)
    envs = ["env1", "env1", "env2"]

    metrics = summarize_stem_diagnostics(diagnostics, keypoints, envs)

    assert metrics["gate_mean"] == pytest.approx(0.25)
    assert metrics["gate_std"] == pytest.approx(0.0)
    assert "background_env_mean_gap" in metrics
    assert "raw_feature_pose_distance_corr" in metrics
    assert "residual_feature_pose_distance_corr" in metrics
    assert "fused_feature_pose_distance_corr" in metrics
