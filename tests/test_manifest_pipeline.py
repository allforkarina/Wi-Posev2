from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.memmap_dataset import MemmapDataset  # noqa: E402
from data.split_manifest import (  # noqa: E402
    DatasetMetadata,
    build_split_arrays,
    compute_source_train_normalization,
    load_manifest,
    save_manifest,
)
from dataloader import create_manifest_data_loader  # noqa: E402
from eval import (  # noqa: E402
    build_evaluation_dataset,
    load_checkpoint_model,
    write_evaluation_outputs,
)
from models import WiFlowMLPDecoder, WiFlowModel  # noqa: E402
from train import TrainConfig, parse_args, prepare_training_config  # noqa: E402


FEW_SHOT_SPECS = {
    "env2_fewshot_8": 2,
    "env2_fewshot_12": 3,
    "env2_fewshot_60": 15,
    "env2_fewshot_120": 30,
}


def _write_dataset(root: Path, frames_per_group: int = 64) -> DatasetMetadata:
    environment: list[str] = []
    subject: list[str] = []
    action: list[str] = []
    for env in ("env1", "env2"):
        for sample in ("S01", "S02"):
            for action_name in ("A01", "A02"):
                environment.extend([env] * frames_per_group)
                subject.extend([sample] * frames_per_group)
                action.extend([action_name] * frames_per_group)
    metadata = DatasetMetadata(
        environment=np.asarray(environment),
        subject=np.asarray(subject),
        action=np.asarray(action),
    )
    count = len(environment)
    csi = np.empty((count, 64, 3, 114), dtype=np.float32)
    for index in range(count):
        csi[index].fill(index / max(1, count - 1))
    np.save(root / "csi_gminmax.npy", csi)
    np.save(root / "ground_truth.npy", np.zeros((count, 18, 2), dtype=np.float32))
    np.savez(
        root / "meta.npz",
        environment=metadata.environment,
        sample=metadata.subject,
        action=metadata.action,
    )
    return metadata


def _write_manifest(root: Path, metadata: DatasetMetadata) -> Path:
    arrays = build_split_arrays(
        metadata,
        mode="random_frame",
        seed=42,
        few_shot_specs=FEW_SHOT_SPECS,
    )
    normalization = compute_source_train_normalization(root, arrays["env1_train"])
    path = root / "random_frame_seed42.npz"
    save_manifest(
        path,
        arrays,
        dataset_root=root,
        mode="random_frame",
        seed=42,
        block_size=16,
        source_train_normalization=normalization,
    )
    return path


def test_source_normalization_uses_only_explicit_train_indices(tmp_path: Path) -> None:
    values = np.asarray([0.0, 0.25, 0.75, 1.0], dtype=np.float32)
    csi = np.broadcast_to(values[:, None, None, None], (4, 2, 3, 4)).copy()
    np.save(tmp_path / "csi_gminmax.npy", csi)

    lower, upper = compute_source_train_normalization(
        tmp_path,
        np.asarray([1, 2], dtype=np.int64),
        chunk_size=1,
    )

    assert lower == pytest.approx(0.25)
    assert upper == pytest.approx(0.75)


def test_explicit_indices_apply_source_affine_without_clamping(tmp_path: Path) -> None:
    metadata = _write_dataset(tmp_path)
    target_index = len(metadata.environment) - 1
    dataset = MemmapDataset(
        data_dir=tmp_path,
        split="all",
        indices=np.asarray([target_index], dtype=np.int64),
        split_normalization=(0.25, 0.75),
    )

    item = dataset[0]

    assert item["meta"]["frame_idx"] == target_index
    assert torch.allclose(item["csi"], torch.full_like(item["csi"], 1.5))


def test_explicit_indices_are_validated(tmp_path: Path) -> None:
    metadata = _write_dataset(tmp_path)
    with pytest.raises(ValueError, match="out-of-range"):
        MemmapDataset(
            data_dir=tmp_path,
            split="all",
            indices=np.asarray([len(metadata.environment)], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="normalization range"):
        MemmapDataset(
            data_dir=tmp_path,
            split="all",
            indices=np.asarray([0], dtype=np.int64),
            split_normalization=(0.5, 0.5),
        )


def test_manifest_loader_uses_absolute_indices_and_saved_normalization(tmp_path: Path) -> None:
    metadata = _write_dataset(tmp_path)
    manifest_path = _write_manifest(tmp_path, metadata)
    manifest = load_manifest(
        manifest_path,
        tmp_path,
        few_shot_keys=tuple(FEW_SHOT_SPECS),
    )
    loader = create_manifest_data_loader(
        data_dir=tmp_path,
        manifest=manifest,
        key="env1_val",
        batch_size=4,
        shuffle=False,
    )

    batch = next(iter(loader))

    assert batch["csi_amplitude"].shape == (4, 3, 114, 64)
    assert batch["frame_idx"] == manifest.indices("env1_val")[:4].tolist()
    assert loader.dataset.split_normalization == manifest.source_train_normalization


def test_training_cli_accepts_manifest_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "train.py",
        "--mode", "finetune",
        "--dataset-root", "dataset",
        "--split-manifest", "split.npz",
        "--few-shot-key", "env2_fewshot_540",
        "--bone-loss-weight", "0.0",
        "--seed", "7",
        "--device", "cpu",
    ])

    args = parse_args()

    assert args.split_manifest == "split.npz"
    assert args.few_shot_key == "env2_fewshot_540"
    assert args.bone_loss_weight == 0.0
    assert args.seed == 7
    assert args.device == "cpu"


def test_training_config_rejects_mlp_latent_structure_loss() -> None:
    config = TrainConfig(
        dataset_root="dataset",
        mode="source_only",
        decoder_type="mlp",
        latent_structure_loss_weight=0.1,
    )
    with pytest.raises(ValueError, match="MLP decoder cannot use latent-structure"):
        prepare_training_config(config)


def test_training_config_records_resolved_manifest_provenance(tmp_path: Path) -> None:
    metadata = _write_dataset(tmp_path)
    manifest_path = _write_manifest(tmp_path, metadata)
    config = TrainConfig(
        dataset_root=str(tmp_path),
        mode="source_only",
        split_manifest=str(manifest_path),
    )

    prepared, manifest = prepare_training_config(config)

    assert manifest is not None
    assert prepared.split_manifest == str(manifest_path.resolve())
    assert prepared.split_mode == "random_frame"
    assert prepared.manifest_hash == manifest.manifest_hash


def test_evaluation_dataset_uses_requested_manifest_key(tmp_path: Path) -> None:
    metadata = _write_dataset(tmp_path)
    manifest_path = _write_manifest(tmp_path, metadata)
    manifest = load_manifest(
        manifest_path,
        tmp_path,
        few_shot_keys=tuple(FEW_SHOT_SPECS),
    )

    dataset = build_evaluation_dataset(tmp_path, manifest, "env2_test")

    assert np.array_equal(dataset.indices, manifest.indices("env2_test"))
    assert dataset.split_normalization == manifest.source_train_normalization


def test_checkpoint_rebuilds_mlp_and_rejects_manifest_mismatch(tmp_path: Path) -> None:
    model = WiFlowModel(decoder_type="mlp")
    checkpoint_path = tmp_path / "model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "train_config": {
            "decoder_type": "mlp",
            "manifest_hash": "expected-hash",
        },
    }, checkpoint_path)

    loaded = load_checkpoint_model(
        checkpoint_path,
        torch.device("cpu"),
        expected_manifest_hash="expected-hash",
    )

    assert isinstance(loaded.decoder, WiFlowMLPDecoder)
    with pytest.raises(ValueError, match="manifest hash"):
        load_checkpoint_model(
            checkpoint_path,
            torch.device("cpu"),
            expected_manifest_hash="different-hash",
        )


def test_evaluation_outputs_include_one_summary_row(tmp_path: Path) -> None:
    result = {
        "sample_count": 3,
        "overall": {
            "mpjpe": 0.25,
            "bone_error": 0.1,
            "pck_0_1": 0.2,
            "pck_0_2": 0.4,
            "pck_0_3": 0.6,
            "pck_0_4": 0.7,
            "pck_0_5": 0.8,
        },
        "joint_rows": [{"joint_index": 0, "mpjpe": 0.25, "pck_0_2": 0.4}],
        "action_rows": [{"action": "A01", "sample_count": 3, "mpjpe": 0.25, "pck_0_2": 0.4}],
        "environment_rows": [{"environment": "env2", "sample_count": 3, "mpjpe": 0.25, "pck_0_2": 0.4}],
        "diagnostic": {
            "overall": {"overall_var_ratio": 0.5, "overall_mean_pose_dist": 0.2},
            "joint_rows": [{"joint_index": 0, "pred_var": 0.1, "gt_var": 0.2, "var_ratio": 0.5, "mean_pose_dist": 0.2}],
        },
    }

    write_evaluation_outputs(tmp_path, result)

    summary = (tmp_path / "benchmark_summary.csv").read_text(encoding="utf-8")
    assert "sample_count,mpjpe,bone_error,pck_0_1,pck_0_2" in summary
    assert "3,0.25,0.1,0.2,0.4" in summary
