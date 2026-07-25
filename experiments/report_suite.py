"""Declarative delivery experiment matrix.

Identical architectures are trained once and carry combined IDs.  For example,
``AX6_JD3_C3`` is simultaneously the full axial control, full decoder control,
and complete proposed model.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
FULL_MODEL_ID = "AX6_JD3_C3"


@dataclass(frozen=True)
class SuiteConfig:
    dataset_root: Path
    output_root: Path
    split_modes: tuple[str, ...]
    seed: int
    source_epochs: int
    finetune_epochs: int
    batch_size: int
    device: str


@dataclass(frozen=True)
class ExperimentTask:
    experiment_id: str
    local_id: str
    split_mode: str
    phase: str
    command: tuple[str, ...]
    output_dir: Path
    manifest_path: Path
    evaluation_keys: tuple[str, ...]
    benchmark_keys: tuple[str, ...] = ()
    visualization_keys: tuple[str, ...] = ()
    few_shot_key: str | None = None

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / "best_val_mpjpe.pth"


def split_slug(split_mode: str) -> str:
    if split_mode == "random_frame":
        return "random_frame"
    if split_mode == "temporal_block":
        return "temporal_block16"
    raise ValueError(f"Unknown split mode: {split_mode}")


def manifest_path(config: SuiteConfig, split_mode: str) -> Path:
    filename = (
        f"random_frame_seed{config.seed}.npz"
        if split_mode == "random_frame"
        else f"temporal_block16_seed{config.seed}.npz"
    )
    return config.output_root / "manifests" / filename


def _common_train_args(
    config: SuiteConfig,
    manifest: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "train.py"),
        "--dataset-root",
        str(config.dataset_root),
        "--output-dir",
        str(output_dir),
        "--split-manifest",
        str(manifest),
        "--batch-size",
        str(config.batch_size),
        "--device",
        config.device,
        "--seed",
        str(config.seed),
    ]


def _source_variants(split_mode: str) -> tuple[tuple[str, str, str], ...]:
    if split_mode == "temporal_block":
        return ((FULL_MODEL_ID, "spatial_then_temporal", "joint"),)
    return (
        (FULL_MODEL_ID, "spatial_then_temporal", "joint"),
        ("AX0_C2", "none", "joint"),
        ("AX1", "spatial_only", "joint"),
        ("AX2", "temporal_only", "joint"),
        ("AX3", "temporal_then_spatial", "joint"),
        ("AX4", "parallel_sum", "joint"),
        ("AX5", "parallel_concat", "joint"),
        ("JD0_C1", "spatial_then_temporal", "mlp"),
        ("JD1", "spatial_then_temporal", "joint_cross_attention"),
        ("JD2", "spatial_then_temporal", "joint_self_attention"),
        ("JD4", "spatial_then_temporal", "joint_shuffled_graph"),
        ("JD5", "spatial_then_temporal", "joint_identity_graph"),
        ("C0", "none", "mlp"),
    )


def _source_tasks(config: SuiteConfig, split_mode: str) -> list[ExperimentTask]:
    manifest = manifest_path(config, split_mode)
    base_dir = config.output_root / split_slug(split_mode) / "source"
    tasks: list[ExperimentTask] = []
    for local_id, axial_mode, decoder_type in _source_variants(split_mode):
        output_dir = base_dir / local_id.lower()
        command = _common_train_args(config, manifest, output_dir) + [
            "--mode",
            "source_only",
            "--source-envs",
            "env1",
            "--epochs",
            str(config.source_epochs),
            "--axial-mode",
            axial_mode,
            "--decoder-type",
            decoder_type,
            "--bone-loss-weight",
            "0.5",
        ]
        visualization_keys: tuple[str, ...] = ()
        if config.seed == 42 and split_mode == "random_frame":
            if local_id == FULL_MODEL_ID:
                visualization_keys = ("env1_test", "env2_test")
            elif local_id in {"AX0_C2", "JD0_C1", "C0"}:
                visualization_keys = ("env1_test",)
        tasks.append(ExperimentTask(
            experiment_id=f"{split_slug(split_mode)}_{local_id}",
            local_id=local_id,
            split_mode=split_mode,
            phase="source",
            command=tuple(command),
            output_dir=output_dir,
            manifest_path=manifest,
            evaluation_keys=("env1_val", "env1_test", "env2_test"),
            benchmark_keys=(
                ("env1_test", "env2_test")
                if local_id == FULL_MODEL_ID else ()
            ),
            visualization_keys=visualization_keys,
        ))
    return tasks


def _finetune_tasks(config: SuiteConfig, split_mode: str) -> list[ExperimentTask]:
    if split_mode != "random_frame":
        return []
    manifest = manifest_path(config, split_mode)
    split_dir = config.output_root / split_slug(split_mode)
    source_checkpoint = (
        split_dir / "source" / FULL_MODEL_ID.lower() / "best_val_mpjpe.pth"
    )
    tasks: list[ExperimentTask] = []
    for local_id, few_shot_key in (
        ("FT540", "env2_fewshot_540"),
        ("FT810", "env2_fewshot_810"),
        ("FT4050", "env2_fewshot_4050"),
        ("FT8100", "env2_fewshot_8100"),
    ):
        output_dir = split_dir / "finetune" / local_id.lower()
        command = _common_train_args(config, manifest, output_dir) + [
            "--mode",
            "finetune",
            "--source-envs",
            "env1",
            "--target-envs",
            "env2",
            "--finetune-from",
            str(source_checkpoint),
            "--few-shot-key",
            few_shot_key,
            "--trainable-groups",
            "full",
            "--epochs",
            str(config.finetune_epochs),
            "--axial-mode",
            "spatial_then_temporal",
            "--decoder-type",
            "joint",
            "--bone-loss-weight",
            "0.5",
        ]
        tasks.append(ExperimentTask(
            experiment_id=f"{split_slug(split_mode)}_{local_id}",
            local_id=local_id,
            split_mode=split_mode,
            phase="finetune",
            command=tuple(command),
            output_dir=output_dir,
            manifest_path=manifest,
            evaluation_keys=("env2_val", "env2_test", "env1_test"),
            benchmark_keys=("env2_test",) if local_id == "FT8100" else (),
            visualization_keys=(
                ("env2_test",)
                if config.seed == 42 and local_id == "FT8100" else ()
            ),
            few_shot_key=few_shot_key,
        ))
    return tasks


def build_training_tasks(config: SuiteConfig) -> list[ExperimentTask]:
    tasks: list[ExperimentTask] = []
    for split_mode in config.split_modes:
        tasks.extend(_source_tasks(config, split_mode))
        tasks.extend(_finetune_tasks(config, split_mode))
    return tasks


def select_trainable_group(rows: Sequence[Mapping[str, str]]) -> str:
    """Retained helper for compatibility with archived report tooling."""
    if not rows:
        raise ValueError("At least one validation row is required")
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["mpjpe"]),
            -float(row["pck_0_2"]),
            row["experiment_id"],
        ),
    )
    return str(ranked[0]["trainable_group"])


def is_task_complete(task: ExperimentTask, expected_manifest_hash: str) -> bool:
    marker_path = task.output_dir / "completed.json"
    if not marker_path.is_file() or not task.checkpoint_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") != "completed":
            return False
        if marker.get("manifest_hash") != expected_manifest_hash:
            return False
        if Path(marker.get("checkpoint_path", "")).resolve() != task.checkpoint_path.resolve():
            return False
        checkpoint = torch.load(task.checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
            return False
        train_config = checkpoint.get("train_config")
        return (
            isinstance(train_config, Mapping)
            and train_config.get("manifest_hash") == expected_manifest_hash
        )
    except Exception:
        return False
