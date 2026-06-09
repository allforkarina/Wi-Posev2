from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import load_checkpoint_model  # noqa: E402
from models import (  # noqa: E402
    CSI_INPUT_CALIBRATION_TYPES,
    AntennaSubcarrierAffineCalibration,
    DynamicAntennaSubcarrierCalibration,
    WiFlowModel,
)
from train import apply_trainable_groups, load_finetune_state_dict  # noqa: E402


@pytest.fixture()
def temp_dir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="input_calibration_", dir=Path.cwd()))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _trainable_names(model: WiFlowModel) -> set[str]:
    return {name for name, param in model.named_parameters() if param.requires_grad}


def test_antenna_subcarrier_affine_calibration_starts_as_identity() -> None:
    layer = AntennaSubcarrierAffineCalibration(num_antennas=3, num_subcarriers=114)
    csi = torch.randn(2, 3, 114, 64)

    calibrated = layer(csi)

    assert CSI_INPUT_CALIBRATION_TYPES == (
        "none",
        "antenna_subcarrier_affine",
        "antenna_subcarrier_dynamic",
    )
    assert calibrated.shape == csi.shape
    assert torch.allclose(calibrated, csi)
    assert layer.log_scale.shape == (1, 3, 114, 1)
    assert layer.bias.shape == (1, 3, 114, 1)


def test_antenna_subcarrier_affine_calibration_applies_per_frequency_response() -> None:
    layer = AntennaSubcarrierAffineCalibration(num_antennas=3, num_subcarriers=114)
    csi = torch.ones(1, 3, 114, 2)
    with torch.no_grad():
        layer.log_scale[:, 1, 10, :] = torch.log(torch.tensor(2.0))
        layer.bias[:, 1, 10, :] = 0.25

    calibrated = layer(csi)

    assert float(calibrated[:, 1, 10].mean().detach()) == pytest.approx(2.25)
    assert float(calibrated[:, 0, 10].mean().detach()) == pytest.approx(1.0)


def test_wiflow_model_applies_input_calibration_before_encoder() -> None:
    model = WiFlowModel(input_calibration="antenna_subcarrier_affine")
    csi = torch.randn(1, 3, 114, 64)

    with torch.no_grad():
        prediction = model(csi)

    assert model.input_calibration_type == "antenna_subcarrier_affine"
    assert prediction.shape == (1, 18, 2)


def test_dynamic_antenna_subcarrier_calibration_starts_as_identity() -> None:
    layer = DynamicAntennaSubcarrierCalibration(
        num_antennas=3,
        num_subcarriers=114,
        basis_count=4,
        hidden_dim=8,
    )
    csi = torch.randn(2, 3, 114, 64)

    calibrated = layer(csi)

    assert calibrated.shape == csi.shape
    assert torch.allclose(calibrated, csi)
    assert layer.log_scale_basis.shape == (4, 1, 3, 114, 1)
    assert layer.bias_basis.shape == (4, 1, 3, 114, 1)
    assert layer.latest_basis_weights.shape == (2, 4)
    assert torch.allclose(layer.latest_basis_weights, torch.zeros_like(layer.latest_basis_weights))


def test_dynamic_antenna_subcarrier_calibration_can_adapt_per_sample() -> None:
    layer = DynamicAntennaSubcarrierCalibration(
        num_antennas=3,
        num_subcarriers=114,
        basis_count=1,
        hidden_dim=1,
    )
    csi = torch.ones(2, 3, 114, 2)
    csi[1] = 2.0
    with torch.no_grad():
        layer.log_scale_basis[0].fill_(torch.log(torch.tensor(2.0)))
        layer.bias_basis.zero_()
        layer.input_projection.weight.zero_()
        layer.input_projection.bias.zero_()
        layer.input_projection.weight[0, 0] = 1.0
        layer.output_projection.weight.zero_()
        layer.output_projection.bias.zero_()
        layer.output_projection.weight[0, 0] = 1.0

    calibrated = layer(csi)

    assert float(calibrated[0].mean().detach()) == pytest.approx(2.0)
    assert float(calibrated[1].mean().detach()) == pytest.approx(8.0)
    assert float(layer.latest_basis_weights[1, 0]) > float(layer.latest_basis_weights[0, 0])


def test_wiflow_model_accepts_dynamic_input_calibration() -> None:
    model = WiFlowModel(input_calibration="antenna_subcarrier_dynamic")
    csi = torch.randn(1, 3, 114, 64)

    with torch.no_grad():
        prediction = model(csi)

    assert model.input_calibration_type == "antenna_subcarrier_dynamic"
    assert isinstance(model.input_calibration, DynamicAntennaSubcarrierCalibration)
    assert prediction.shape == (1, 18, 2)


def test_eval_rebuilds_input_calibration_from_checkpoint_config(temp_dir: Path) -> None:
    checkpoint_path = temp_dir / "input_calibration.pth"
    model = WiFlowModel(input_calibration="antenna_subcarrier_affine")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": {
                "axial_mode": "spatial_then_temporal",
                "decoder_type": "joint",
                "csi_feature_mode": "raw",
                "spatial_stem_type": "baseline",
                "background_kernel_size": 9,
                "input_calibration": "antenna_subcarrier_affine",
            },
        },
        checkpoint_path,
    )

    loaded = load_checkpoint_model(checkpoint_path, torch.device("cpu"))

    assert loaded.input_calibration_type == "antenna_subcarrier_affine"
    assert isinstance(loaded.input_calibration, AntennaSubcarrierAffineCalibration)


def test_eval_rebuilds_dynamic_input_calibration_from_checkpoint_config(temp_dir: Path) -> None:
    checkpoint_path = temp_dir / "dynamic_input_calibration.pth"
    model = WiFlowModel(input_calibration="antenna_subcarrier_dynamic")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": {
                "axial_mode": "spatial_then_temporal",
                "decoder_type": "joint",
                "csi_feature_mode": "raw",
                "spatial_stem_type": "baseline",
                "background_kernel_size": 9,
                "input_calibration": "antenna_subcarrier_dynamic",
            },
        },
        checkpoint_path,
    )

    loaded = load_checkpoint_model(checkpoint_path, torch.device("cpu"))

    assert loaded.input_calibration_type == "antenna_subcarrier_dynamic"
    assert isinstance(loaded.input_calibration, DynamicAntennaSubcarrierCalibration)


def test_input_calibration_trainable_group_only_updates_calibration_parameters() -> None:
    model = WiFlowModel(input_calibration="antenna_subcarrier_affine")

    trainable_count = apply_trainable_groups(model, ("input_calibration",))
    names = _trainable_names(model)

    assert trainable_count == 3 * 114 * 2
    assert names == {"input_calibration.log_scale", "input_calibration.bias"}


def test_dynamic_input_calibration_trainable_group_only_updates_calibration_parameters() -> None:
    model = WiFlowModel(input_calibration="antenna_subcarrier_dynamic")

    trainable_count = apply_trainable_groups(model, ("input_calibration",))
    names = _trainable_names(model)

    assert trainable_count > 3 * 114 * 2
    assert names
    assert all(name.startswith("input_calibration.") for name in names)
    assert "input_calibration.log_scale_basis" in names
    assert "input_calibration.output_projection.weight" in names


def test_finetune_loader_accepts_missing_new_calibration_parameters() -> None:
    source_model = WiFlowModel()
    target_model = WiFlowModel(input_calibration="antenna_subcarrier_affine")

    incompatible = load_finetune_state_dict(target_model, source_model.state_dict())

    assert set(incompatible.missing_keys) == {
        "input_calibration.log_scale",
        "input_calibration.bias",
    }
    assert incompatible.unexpected_keys == []


def test_finetune_loader_accepts_missing_dynamic_calibration_parameters() -> None:
    source_model = WiFlowModel()
    target_model = WiFlowModel(input_calibration="antenna_subcarrier_dynamic")

    incompatible = load_finetune_state_dict(target_model, source_model.state_dict())

    assert incompatible.missing_keys
    assert all(name.startswith("input_calibration.") for name in incompatible.missing_keys)
    assert incompatible.unexpected_keys == []
