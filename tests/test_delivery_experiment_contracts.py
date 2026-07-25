from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from data.pose_schema import (
    CANONICAL_BONE_EDGES,
    MAPPED_18_TO_RAW_17,
    map_raw17_to_project18,
)
from experiments.report_suite import SuiteConfig, build_training_tasks
from eval import _array_metrics
from models.skeleton import OPENPOSE_BONE_EDGES
from models.wiflow_axial_encoder import AXIAL_ENCODER_MODES, WiFlowAxialEncoder
from models.wiflow_joint_decoder import WiFlowJointDecoder
from scripts.data.build_memmap import build_dataset
from train import compute_torso_scale


def test_pose_mapping_preserves_coordinates_and_leg_midpoint() -> None:
    raw = np.arange(17 * 2, dtype=np.float32).reshape(17, 2)
    raw[0] = 0.0
    mapped = map_raw17_to_project18(raw)

    assert mapped.shape == (18, 2)
    assert np.array_equal(mapped[0], raw[0])
    for mapped_index, raw_index in MAPPED_18_TO_RAW_17.items():
        assert np.array_equal(mapped[mapped_index], raw[raw_index])
    assert np.array_equal(mapped[1], (raw[5] + raw[6]) / 2.0)
    assert OPENPOSE_BONE_EDGES == CANONICAL_BONE_EDGES


def test_torso_scale_uses_custom_cross_body_diagonals() -> None:
    target = torch.zeros(1, 18, 2)
    target[0, 9] = torch.tensor([0.0, 0.0])
    target[0, 16] = torch.tensor([3.0, 4.0])
    target[0, 11] = torch.tensor([0.0, 0.0])
    target[0, 15] = torch.tensor([0.0, 2.0])

    assert torch.allclose(compute_torso_scale(target), torch.tensor([3.5]))


def test_axial_and_decoder_ablation_shapes() -> None:
    axial_input = torch.randn(2, 128, 3, 4)
    for mode in AXIAL_ENCODER_MODES:
        assert WiFlowAxialEncoder(mode)(axial_input).shape == (2, 256, 3, 4)

    decoder_input = torch.randn(2, 256, 3, 4)
    variants = (
        WiFlowJointDecoder(use_graph=False, use_joint_attention=False),
        WiFlowJointDecoder(use_graph=False, use_joint_attention=True),
        WiFlowJointDecoder(adjacency_variant="canonical"),
        WiFlowJointDecoder(adjacency_variant="identity"),
        WiFlowJointDecoder(adjacency_variant="shuffled"),
    )
    for decoder in variants:
        assert decoder(decoder_input).shape == (2, 18, 2)


def test_extended_metrics_are_zero_for_an_exact_prediction() -> None:
    target = np.random.default_rng(42).normal(size=(4, 18, 2)).astype(np.float32)
    metrics = _array_metrics(target.copy(), target)

    for name in (
        "mpjpe",
        "median_joint_error",
        "p95_joint_error",
        "coordinate_rmse",
        "n_mpjpe",
        "root_relative_mpjpe",
        "pa_mpjpe",
        "bone_error",
        "relative_bone_length_error",
        "bone_direction_error_deg",
        "symmetry_error",
        "invalid_skeleton_rate",
    ):
        assert abs(metrics[name]) < 1e-5
    assert metrics["pck_0_2"] == 1.0
    assert metrics["pck_auc_0_5"] > 0.99


def test_delivery_matrix_trains_each_unique_configuration_once(tmp_path: Path) -> None:
    config = SuiteConfig(
        dataset_root=tmp_path / "dataset",
        output_root=tmp_path / "outputs",
        split_modes=("random_frame", "temporal_block"),
        seed=42,
        source_epochs=50,
        finetune_epochs=30,
        batch_size=64,
        device="cuda",
    )
    tasks = build_training_tasks(config)

    assert len(tasks) == 18
    assert len({task.experiment_id for task in tasks}) == len(tasks)
    assert all(task.checkpoint_path.name == "best_val_mpjpe.pth" for task in tasks)
    assert sum(task.phase == "source" for task in tasks) == 14
    assert sum(task.phase == "finetune" for task in tasks) == 4


def test_memmap_builder_preserves_raw_gt_and_is_resumable(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    wifi_dir = raw_root / "A01" / "S01" / "wifi-csi"
    wifi_dir.mkdir(parents=True)
    for frame in (1, 2):
        sio.savemat(
            wifi_dir / f"frame{frame:03d}.mat",
            {"CSIamp": np.full((3, 114, 10), frame, dtype=np.float32)},
        )
    gt_root = tmp_path / "gt"
    gt_root.mkdir()
    raw_gt = np.arange(2 * 17 * 3, dtype=np.float32).reshape(2, 17, 3) / 10.0
    raw_gt[0, 0, :2] = 0.0
    raw_gt[1, 10, 1] = 4.2
    np.save(gt_root / "E01_S01_A01.npy", raw_gt)
    output = tmp_path / "memmap"
    args = Namespace(
        src=raw_root,
        dst=output,
        gt_dir=gt_root,
        train_subjects=["S01"],
        workers=1,
        chunk_size=1,
        extra_normalizations=(),
        resume=False,
    )

    build_dataset(args)

    mapped = np.load(output / "ground_truth.npy")
    assert np.array_equal(mapped, map_raw17_to_project18(raw_gt[..., :2]))
    assert float(mapped.max()) > 4.0
    assert np.array_equal(mapped[0, 0], np.zeros(2, dtype=np.float32))
    assert not (output / "csi_raw.build.npy").exists()
    assert (output / "build_complete.json").is_file()

    args.resume = True
    build_dataset(args)
