from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import NUM_OPENPOSE_KEYPOINTS, WiFlowJointDecoder, WiFlowModel  # noqa: E402
from train import joint_latent_structure_loss  # noqa: E402


def test_joint_decoder_can_return_latent_features_without_changing_default_output() -> None:
    decoder = WiFlowJointDecoder()
    features = torch.randn(2, 256, 29, 16)

    coordinates = decoder(features)
    coordinates_with_latents, decoder_features = decoder(features, return_features=True)

    assert coordinates.shape == (2, NUM_OPENPOSE_KEYPOINTS, 2)
    assert coordinates_with_latents.shape == coordinates.shape
    assert decoder_features.shape == (2, NUM_OPENPOSE_KEYPOINTS, 256)


def test_wiflow_model_forwards_decoder_latent_features() -> None:
    model = WiFlowModel()
    csi = torch.randn(1, 3, 114, 64)

    with torch.no_grad():
        coordinates, decoder_features = model(csi, return_decoder_features=True)

    assert coordinates.shape == (1, NUM_OPENPOSE_KEYPOINTS, 2)
    assert decoder_features.shape == (1, NUM_OPENPOSE_KEYPOINTS, 256)


def test_joint_latent_structure_loss_is_differentiable() -> None:
    decoder_features = torch.randn(3, NUM_OPENPOSE_KEYPOINTS, 16, requires_grad=True)
    target = torch.randn(3, NUM_OPENPOSE_KEYPOINTS, 2)

    loss = joint_latent_structure_loss(decoder_features, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert decoder_features.grad is not None
    assert torch.isfinite(decoder_features.grad).all()


def test_joint_latent_structure_loss_validates_shapes() -> None:
    decoder_features = torch.randn(2, NUM_OPENPOSE_KEYPOINTS, 16)
    target = torch.randn(2, NUM_OPENPOSE_KEYPOINTS - 1, 2)

    with pytest.raises(ValueError, match="share batch and joint dimensions"):
        joint_latent_structure_loss(decoder_features, target)
