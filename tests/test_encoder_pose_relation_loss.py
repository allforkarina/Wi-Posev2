from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import WiFlowModel  # noqa: E402
from train import compute_losses, pose_feature_relation_loss  # noqa: E402


def test_wiflow_model_exposes_axial_encoder_features() -> None:
    model = WiFlowModel()
    csi = torch.randn(2, 3, 114, 64)

    with torch.no_grad():
        encoder_features = model.encode_features(csi)
        prediction = model(csi)

    assert encoder_features.shape == (2, 256, 29, 16)
    assert prediction.shape == (2, 18, 2)


def test_pose_feature_relation_loss_is_differentiable() -> None:
    encoder_features = torch.randn(4, 8, 3, 2, requires_grad=True)
    target = torch.randn(4, 18, 2)

    loss = pose_feature_relation_loss(encoder_features, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert encoder_features.grad is not None
    assert torch.isfinite(encoder_features.grad).all()


def test_pose_feature_relation_loss_returns_zero_for_single_sample_batch() -> None:
    encoder_features = torch.randn(1, 8, 3, 2, requires_grad=True)
    target = torch.randn(1, 18, 2)

    loss = pose_feature_relation_loss(encoder_features, target)
    loss.backward()

    assert float(loss.detach()) == 0.0
    assert encoder_features.grad is not None


def test_compute_losses_includes_encoder_relation_loss_when_weighted() -> None:
    prediction = torch.randn(4, 18, 2)
    target = torch.randn(4, 18, 2)
    encoder_features = torch.randn(4, 8, 3, 2)

    losses = compute_losses(
        prediction,
        target,
        encoder_features=encoder_features,
        encoder_relation_loss_weight=0.5,
    )

    expected_total = (
        losses["coord_loss"]
        + 0.5 * losses["bone_loss"]
        + 0.5 * losses["encoder_relation_loss"]
    )
    assert "encoder_relation_loss" in losses
    assert torch.allclose(losses["loss"], expected_total)
