from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import DECODER_TYPES, WiFlowMLPDecoder, WiFlowModel  # noqa: E402


def test_mlp_decoder_shape_parameter_budget_and_backward() -> None:
    decoder = WiFlowMLPDecoder()
    features = torch.randn(2, 256, 29, 16, requires_grad=True)

    coordinates = decoder(features)

    assert coordinates.shape == (2, 18, 2)
    assert sum(parameter.numel() for parameter in decoder.parameters()) == 2_005_540
    coordinates.sum().backward()
    assert features.grad is not None


def test_mlp_decoder_rejects_joint_latent_request() -> None:
    decoder = WiFlowMLPDecoder()
    with pytest.raises(ValueError, match="does not expose joint latent features"):
        decoder(torch.randn(1, 256, 29, 16), return_features=True)


def test_mlp_decoder_validates_encoder_shape() -> None:
    decoder = WiFlowMLPDecoder()
    with pytest.raises(ValueError, match=r"\[B, 256, H, W\]"):
        decoder(torch.randn(1, 128, 29, 16))


def test_wiflow_model_supports_all_three_decoder_types() -> None:
    assert DECODER_TYPES == ("mlp", "joint", "hierarchical")
    model = WiFlowModel(decoder_type="mlp")

    with torch.no_grad():
        output = model(torch.randn(1, 3, 114, 64))

    assert isinstance(model.decoder, WiFlowMLPDecoder)
    assert output.shape == (1, 18, 2)
