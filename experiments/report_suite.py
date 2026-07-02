from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import torch


SELECTED_GROUP_TOKEN = "{selected_trainable_group}"
ROOT = Path(__file__).resolve().parents[1]


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
    trainable_group: str | None = None
    few_shot_key: str | None = None

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / "best_val_pck_0_2.pth"


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


def _common_train_args(config: SuiteConfig, manifest: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "train.py"),
        "--dataset-root", str(config.dataset_root),
        "--output-dir", str(output_dir),
        "--split-manifest", str(manifest),
        "--batch-size", str(config.batch_size),
        "--device", config.device,
        "--seed", str(config.seed),
    ]


def _source_tasks(config: SuiteConfig, split_mode: str) -> list[ExperimentTask]:
    manifest = manifest_path(config, split_mode)
    base_dir = config.output_root / split_slug(split_mode) / "source"
    variants = (
        ("A1", "spatial_then_temporal", "joint", 0.5),
        ("A2", "temporal_then_spatial", "joint", 0.5),
        ("A3", "parallel_sum", "joint", 0.5),
        ("A4", "parallel_concat", "joint", 0.5),
        ("D1", "spatial_then_temporal", "mlp", 0.5),
        ("D3", "spatial_then_temporal", "hierarchical", 0.5),
        ("B1", "spatial_then_temporal", "joint", 0.0),
    )
    tasks: list[ExperimentTask] = []
    for local_id, axial_mode, decoder_type, bone_weight in variants:
        output_dir = base_dir / local_id.lower()
        command = _common_train_args(config, manifest, output_dir) + [
            "--mode", "source_only",
            "--source-envs", "env1",
            "--epochs", str(config.source_epochs),
            "--axial-mode", axial_mode,
            "--decoder-type", decoder_type,
            "--bone-loss-weight", str(bone_weight),
        ]
        tasks.append(ExperimentTask(
            experiment_id=f"{split_slug(split_mode)}_{local_id}",
            local_id=local_id,
            split_mode=split_mode,
            phase="source",
            command=tuple(command),
            output_dir=output_dir,
            manifest_path=manifest,
        ))
    return tasks


def _finetune_tasks(config: SuiteConfig, split_mode: str) -> list[ExperimentTask]:
    manifest = manifest_path(config, split_mode)
    split_dir = config.output_root / split_slug(split_mode)
    source_checkpoint = split_dir / "source" / "a1" / "best_val_pck_0_2.pth"
    groups = (
        ("F1", "spatial_encoder"),
        ("F2", "axial_encoder"),
        ("F3", "encoder"),
        ("F4", "decoder"),
        ("F5", "full"),
    )
    tasks: list[ExperimentTask] = []
    for local_id, group in groups:
        output_dir = split_dir / "finetune_540" / local_id.lower()
        command = _common_train_args(config, manifest, output_dir) + [
            "--mode", "finetune",
            "--target-envs", "env2",
            "--finetune-from", str(source_checkpoint),
            "--few-shot-key", "env2_fewshot_540",
            "--trainable-groups", group,
            "--epochs", str(config.finetune_epochs),
            "--axial-mode", "spatial_then_temporal",
            "--decoder-type", "joint",
            "--bone-loss-weight", "0.5",
        ]
        tasks.append(ExperimentTask(
            experiment_id=f"{split_slug(split_mode)}_{local_id}",
            local_id=local_id,
            split_mode=split_mode,
            phase="finetune_540",
            command=tuple(command),
            output_dir=output_dir,
            manifest_path=manifest,
            trainable_group=group,
            few_shot_key="env2_fewshot_540",
        ))

    for local_id, few_shot_key in (
        ("V2", "env2_fewshot_810"),
        ("V3", "env2_fewshot_4050"),
        ("V4", "env2_fewshot_8100"),
    ):
        output_dir = split_dir / "finetune_scale" / local_id.lower()
        command = _common_train_args(config, manifest, output_dir) + [
            "--mode", "finetune",
            "--target-envs", "env2",
            "--finetune-from", str(source_checkpoint),
            "--few-shot-key", few_shot_key,
            "--trainable-groups", SELECTED_GROUP_TOKEN,
            "--epochs", str(config.finetune_epochs),
            "--axial-mode", "spatial_then_temporal",
            "--decoder-type", "joint",
            "--bone-loss-weight", "0.5",
        ]
        tasks.append(ExperimentTask(
            experiment_id=f"{split_slug(split_mode)}_{local_id}",
            local_id=local_id,
            split_mode=split_mode,
            phase="finetune_scale",
            command=tuple(command),
            output_dir=output_dir,
            manifest_path=manifest,
            trainable_group=SELECTED_GROUP_TOKEN,
            few_shot_key=few_shot_key,
        ))
    return tasks


def build_training_tasks(config: SuiteConfig) -> list[ExperimentTask]:
    tasks: list[ExperimentTask] = []
    for split_mode in config.split_modes:
        tasks.extend(_source_tasks(config, split_mode))
        tasks.extend(_finetune_tasks(config, split_mode))
    return tasks


def resolve_scale_task(task: ExperimentTask, trainable_group: str) -> ExperimentTask:
    if SELECTED_GROUP_TOKEN not in task.command:
        return task
    command = tuple(
        trainable_group if value == SELECTED_GROUP_TOKEN else value
        for value in task.command
    )
    return replace(task, command=command, trainable_group=trainable_group)


def select_trainable_group(rows: Sequence[Mapping[str, str]]) -> str:
    if len(rows) != 5:
        raise ValueError("Layer-wise selection requires exactly five validation rows")
    required_groups = {"spatial_encoder", "axial_encoder", "encoder", "decoder", "full"}
    if {row.get("trainable_group") for row in rows} != required_groups:
        raise ValueError("Validation rows must cover all five trainable groups")
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["pck_0_2"]),
            float(row["mpjpe"]),
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
