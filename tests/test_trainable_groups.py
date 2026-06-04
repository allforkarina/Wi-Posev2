from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import WiFlowModel
from train import TRAINABLE_GROUPS, apply_trainable_groups


def _trainable_names(model: WiFlowModel) -> set[str]:
    return {name for name, param in model.named_parameters() if param.requires_grad}


def test_encoder_group_trains_only_spatial_and_axial_encoders() -> None:
    model = WiFlowModel()

    trainable_count = apply_trainable_groups(model, ("encoder",))
    names = _trainable_names(model)

    assert trainable_count > 0
    assert names
    assert all(
        name.startswith(("spatial_encoder.", "axial_encoder."))
        for name in names
    )
    assert not any(name.startswith("decoder.") for name in names)


def test_decoder_group_trains_only_decoder() -> None:
    model = WiFlowModel()

    trainable_count = apply_trainable_groups(model, ("decoder",))
    names = _trainable_names(model)

    assert trainable_count > 0
    assert names
    assert all(name.startswith("decoder.") for name in names)
    assert not any(
        name.startswith(("spatial_encoder.", "axial_encoder."))
        for name in names
    )


def test_full_group_trains_every_parameter() -> None:
    model = WiFlowModel()

    trainable_count = apply_trainable_groups(model, ("full",))

    assert trainable_count == sum(param.numel() for param in model.parameters())
    assert all(param.requires_grad for param in model.parameters())


def test_multiple_groups_train_union_without_decoder_leakage() -> None:
    model = WiFlowModel()

    apply_trainable_groups(model, ("spatial_encoder", "axial_attention"))
    names = _trainable_names(model)

    assert any(name.startswith("spatial_encoder.") for name in names)
    assert any(name.startswith("axial_encoder.spatial_attention.") for name in names)
    assert any(name.startswith("axial_encoder.temporal_attention.") for name in names)
    assert not any(name.startswith("decoder.") for name in names)
    assert not any(name.startswith("axial_encoder.channel_projection.") for name in names)


def test_norms_group_includes_batchnorm_parameters_inside_sequential() -> None:
    model = WiFlowModel()

    apply_trainable_groups(model, ("norms",))
    names = _trainable_names(model)

    assert "spatial_encoder.antenna_mixer.1.weight" in names
    assert "spatial_encoder.feature_stem.1.weight" in names
    assert "axial_encoder.spatial_norm.weight" in names
    assert "decoder.attention_norm.weight" in names
    assert not any(name.endswith("main_path.0.weight") for name in names)


def test_unknown_trainable_group_reports_valid_options() -> None:
    model = WiFlowModel()

    with pytest.raises(ValueError) as exc_info:
        apply_trainable_groups(model, ("encoder", "not_a_group"))

    message = str(exc_info.value)
    assert "Unknown trainable group" in message
    for group in TRAINABLE_GROUPS:
        assert group in message
