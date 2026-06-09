from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import load_checkpoint_model  # noqa: E402
from models import NUM_OPENPOSE_KEYPOINTS, WiFlowModel  # noqa: E402
from train import TRAINABLE_GROUPS, apply_trainable_groups, load_finetune_state_dict  # noqa: E402


@pytest.fixture()
def temp_dir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="wrist_refinement_", dir=Path.cwd()))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _trainable_names(model: WiFlowModel) -> set[str]:
    return {name for name, param in model.named_parameters() if param.requires_grad}


def test_wrist_refinement_initially_preserves_decoder_coordinates() -> None:
    torch.manual_seed(7)
    model = WiFlowModel(wrist_refinement=True)
    model.eval()
    features = torch.randn(2, 256, 29, 16)

    with torch.no_grad():
        base_coordinates, decoder_features = model.decoder(features, return_features=True)
        refined_coordinates = model.decode_features(features)
        refined_with_features, returned_features = model.decode_features(
            features,
            return_decoder_features=True,
        )

    assert refined_coordinates.shape == (2, NUM_OPENPOSE_KEYPOINTS, 2)
    assert torch.allclose(refined_coordinates, base_coordinates)
    assert torch.allclose(refined_with_features, base_coordinates)
    assert torch.allclose(returned_features, decoder_features)


def test_wrist_refinement_changes_only_custom_wrist_joints_after_training_signal() -> None:
    model = WiFlowModel(wrist_refinement=True)
    model.eval()
    features = torch.randn(2, 256, 29, 16)

    with torch.no_grad():
        model.wrist_refiner.output_layer.bias.copy_(torch.tensor([0.1, -0.2, -0.3, 0.4]))
        base_coordinates = model.decoder(features)
        refined_coordinates = model.decode_features(features)

    changed = torch.linalg.vector_norm(refined_coordinates - base_coordinates, dim=-1) > 0
    expected = torch.zeros(2, NUM_OPENPOSE_KEYPOINTS, dtype=torch.bool)
    expected[:, [10, 12]] = True
    assert torch.equal(changed, expected)


def test_wrist_refiner_trainable_group_only_updates_refiner_parameters() -> None:
    model = WiFlowModel(wrist_refinement=True)

    trainable_count = apply_trainable_groups(model, ("wrist_refiner",))
    names = _trainable_names(model)

    assert "wrist_refiner" in TRAINABLE_GROUPS
    assert trainable_count > 0
    assert names
    assert all(name.startswith("wrist_refiner.") for name in names)


def test_old_checkpoint_can_initialize_wrist_refinement_parameters() -> None:
    source_model = WiFlowModel()
    target_model = WiFlowModel(wrist_refinement=True)

    incompatible = load_finetune_state_dict(target_model, source_model.state_dict())

    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(name.startswith("wrist_refiner.") for name in incompatible.missing_keys)


def test_eval_rebuilds_wrist_refinement_from_checkpoint_config(temp_dir: Path) -> None:
    checkpoint_path = temp_dir / "wrist_refinement.pth"
    model = WiFlowModel(wrist_refinement=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": {
                "axial_mode": "spatial_then_temporal",
                "decoder_type": "joint",
                "csi_feature_mode": "raw",
                "spatial_stem_type": "baseline",
                "background_kernel_size": 9,
                "input_calibration": "none",
                "wrist_refinement": True,
            },
        },
        checkpoint_path,
    )

    loaded = load_checkpoint_model(checkpoint_path, torch.device("cpu"))

    assert loaded.wrist_refinement is True
    assert hasattr(loaded, "wrist_refiner")
