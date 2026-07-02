from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.report_suite import (  # noqa: E402
    SELECTED_GROUP_TOKEN,
    SuiteConfig,
    build_training_tasks,
    is_task_complete,
    resolve_scale_task,
    select_trainable_group,
)
import scripts.run_report_experiments as runner  # noqa: E402

run_suite_main = runner.main


def _config(tmp_path: Path) -> SuiteConfig:
    return SuiteConfig(
        dataset_root=tmp_path / "dataset",
        output_root=tmp_path / "output",
        split_modes=("random_frame", "temporal_block"),
        seed=42,
        source_epochs=50,
        finetune_epochs=30,
        batch_size=64,
        device="cuda",
    )


def test_report_suite_contains_exactly_thirty_unique_training_tasks(tmp_path: Path) -> None:
    tasks = build_training_tasks(_config(tmp_path))

    assert len(tasks) == 30
    assert len({task.experiment_id for task in tasks}) == 30
    assert [task.split_mode for task in tasks[:15]] == ["random_frame"] * 15
    assert [task.split_mode for task in tasks[15:]] == ["temporal_block"] * 15
    assert sum(task.phase == "source" for task in tasks) == 14
    assert sum(task.phase == "finetune_540" for task in tasks) == 10
    assert sum(task.phase == "finetune_scale" for task in tasks) == 6
    assert len({task.output_dir for task in tasks}) == 30


def test_source_and_finetune_commands_keep_matched_controls(tmp_path: Path) -> None:
    tasks = build_training_tasks(_config(tmp_path))
    forbidden = {
        "--latent-structure-loss-weight",
        "--encoder-relation-loss-weight",
        "--distal-loss-weight",
        "--wrist-direction-loss-weight",
        "--source-replay-weight",
    }

    for task in tasks:
        command = set(task.command)
        assert not command & forbidden
        assert "--split-manifest" in command
        assert "--seed" in command
        assert "42" in command
        if task.phase.startswith("finetune"):
            assert "--finetune-from" in command
            assert "best_val_pck_0_2.pth" in " ".join(task.command)

    per_split_source_ids = {
        task.local_id
        for task in tasks
        if task.split_mode == "random_frame" and task.phase == "source"
    }
    assert per_split_source_ids == {"A1", "A2", "A3", "A4", "D1", "D3", "B1"}


def test_scale_tasks_resolve_only_the_selected_trainable_group(tmp_path: Path) -> None:
    task = next(
        task
        for task in build_training_tasks(_config(tmp_path))
        if task.local_id == "V2" and task.split_mode == "random_frame"
    )
    assert SELECTED_GROUP_TOKEN in task.command

    resolved = resolve_scale_task(task, "axial_encoder")

    assert SELECTED_GROUP_TOKEN not in resolved.command
    group_position = resolved.command.index("--trainable-groups") + 1
    assert resolved.command[group_position] == "axial_encoder"


def test_validation_selection_uses_pck_then_mpjpe_tie_break() -> None:
    rows = [
        {"experiment_id": "F1", "trainable_group": "spatial_encoder", "pck_0_2": "0.70", "mpjpe": "0.20"},
        {"experiment_id": "F2", "trainable_group": "axial_encoder", "pck_0_2": "0.75", "mpjpe": "0.25"},
        {"experiment_id": "F3", "trainable_group": "encoder", "pck_0_2": "0.75", "mpjpe": "0.18"},
        {"experiment_id": "F4", "trainable_group": "decoder", "pck_0_2": "0.72", "mpjpe": "0.17"},
        {"experiment_id": "F5", "trainable_group": "full", "pck_0_2": "0.71", "mpjpe": "0.16"},
    ]
    assert select_trainable_group(rows) == "encoder"

    with pytest.raises(ValueError, match="exactly five"):
        select_trainable_group(rows[:-1])


def test_resume_requires_marker_loadable_checkpoint_and_matching_hash(tmp_path: Path) -> None:
    task = build_training_tasks(_config(tmp_path))[0]
    task.output_dir.mkdir(parents=True)
    checkpoint = task.checkpoint_path
    torch.save({
        "model_state_dict": {"weight": torch.ones(1)},
        "train_config": {"manifest_hash": "manifest-hash"},
    }, checkpoint)
    (task.output_dir / "completed.json").write_text(json.dumps({
        "status": "completed",
        "checkpoint_path": str(checkpoint),
        "manifest_hash": "manifest-hash",
    }), encoding="utf-8")

    assert is_task_complete(task, "manifest-hash")
    assert not is_task_complete(task, "different-hash")
    (task.output_dir / "completed.json").unlink()
    assert not is_task_complete(task, "manifest-hash")
    (task.output_dir / "completed.json").write_text(json.dumps({
        "status": "completed",
        "checkpoint_path": str(checkpoint),
        "manifest_hash": "manifest-hash",
    }), encoding="utf-8")
    checkpoint.write_text("not a checkpoint", encoding="utf-8")
    assert not is_task_complete(task, "manifest-hash")


def test_dry_run_writes_thirty_planned_registry_rows(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    exit_code = run_suite_main([
        "--dataset-root", str(tmp_path / "dataset"),
        "--output-root", str(output_root),
        "--split-modes", "random_frame", "temporal_block",
        "--seed", "42",
        "--source-epochs", "50",
        "--finetune-epochs", "30",
        "--batch-size", "64",
        "--device", "cuda",
        "--dry-run",
    ])

    assert exit_code == 0
    with (output_root / "experiment_registry.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 30
    assert {row["status"] for row in rows} == {"planned"}
    assert len({row["experiment_id"] for row in rows}) == 30


def test_dry_run_canonicalizes_random_frame_before_temporal_block(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    run_suite_main([
        "--dataset-root", str(tmp_path / "dataset"),
        "--output-root", str(output_root),
        "--split-modes", "temporal_block", "random_frame",
        "--dry-run",
    ])
    with (output_root / "experiment_registry.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["split_mode"] for row in rows[:15]] == ["random_frame"] * 15


def test_postprocess_failure_is_recorded_for_continue_on_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = build_training_tasks(_config(tmp_path))[0]
    registry_path = tmp_path / "registry.csv"
    registry = {task.experiment_id: runner._registry_row(task, status="completed")}

    def fail_command(command: tuple[str, ...]) -> None:
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(runner, "_run_command", fail_command)

    succeeded = runner._run_postprocess(
        task,
        ("python", "eval.py"),
        "evaluation",
        registry_path,
        registry,
    )

    assert not succeeded
    assert registry[task.experiment_id]["status"] == "failed"
    assert "evaluation failed" in registry[task.experiment_id]["failure"]
