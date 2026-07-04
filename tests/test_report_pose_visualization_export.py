from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.pose_viz import save_pose_comparison
from scripts.export_report_pose_visualizations import (
    REPORT_JOBS,
    SampleRecord,
    records_for_job,
    resolve_report_jobs,
    select_one_sample_per_action,
)


def _pose(offset: float = 0.0) -> np.ndarray:
    values = np.linspace(0.1, 0.9, 36, dtype=np.float32).reshape(18, 2)
    return values + offset


@dataclass
class FakeDataset:
    indices: np.ndarray
    _actions: np.ndarray
    _samples: np.ndarray
    _envs: np.ndarray


def _fake_dataset(environment: str) -> FakeDataset:
    return FakeDataset(
        indices=np.asarray([2, 4, 6, 8, 10, 12], dtype=np.int64),
        _actions=np.asarray([
            "unused", "unused", "A01", "unused", "A01", "unused", "A02",
            "unused", "A02", "unused", "A03", "unused", "A03",
        ]),
        _samples=np.asarray([
            "X", "X", "S01", "X", "S02", "X", "S01",
            "X", "S02", "X", "S01", "X", "S02",
        ]),
        _envs=np.asarray([
            "X", "X", environment, "X", environment, "X", environment,
            "X", environment, "X", environment, "X", environment,
        ]),
    )


def test_save_pose_comparison_writes_indexed_png_only(tmp_path: Path) -> None:
    path = save_pose_comparison(
        target=_pose(),
        prediction=_pose(0.01),
        action="A01",
        subject="S01",
        environment="env2",
        output_dir=tmp_path,
        dataset_index=123,
        model_label="Finetune 540",
    )

    assert path == tmp_path / "A01" / "S01_env2_idx123.png"
    assert path.is_file()
    assert not list(tmp_path.rglob("*.pdf"))


def test_save_pose_comparison_preserves_legacy_filename_without_index(
    tmp_path: Path,
) -> None:
    path = save_pose_comparison(
        target=_pose(),
        prediction=_pose(0.01),
        action="A01",
        subject="S01",
        environment="env1",
        output_dir=tmp_path,
    )

    assert path == tmp_path / "A01" / "S01_env1.png"


def test_save_pose_comparison_rejects_non_openpose18_shapes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"shape \[18, 2\]"):
        save_pose_comparison(
            target=np.zeros((17, 2), dtype=np.float32),
            prediction=np.zeros((18, 2), dtype=np.float32),
            action="A01",
            subject="S01",
            environment="env1",
            output_dir=tmp_path,
        )


def test_select_one_sample_per_action_is_deterministic() -> None:
    dataset = _fake_dataset("env2")

    first = select_one_sample_per_action(dataset, seed=42)
    second = select_one_sample_per_action(dataset, seed=42)

    assert first == second
    assert [record.action for record in first] == ["A01", "A02", "A03"]
    assert len({record.dataset_index for record in first}) == 3
    assert {record.environment for record in first} == {"env2"}


def test_report_jobs_resolve_approved_random_frame_paths(tmp_path: Path) -> None:
    resolved = resolve_report_jobs(tmp_path)

    assert [job.output_name for job in resolved] == [
        "source_a1_env1",
        "finetune_540_env2",
        "finetune_810_env2",
        "finetune_4050_env2",
        "finetune_8100_env2",
    ]
    assert resolved[0].manifest_key == "env1_test"
    assert all(job.manifest_key == "env2_test" for job in resolved[1:])
    assert resolved[-1].checkpoint_path == (
        tmp_path / "random_frame/finetune_scale/v4/best_val_pck_0_2.pth"
    )
    assert len(REPORT_JOBS) == 5


def test_all_target_jobs_reuse_the_same_record_sequence(tmp_path: Path) -> None:
    source = (SampleRecord("A01", 1, "S01", "env1"),)
    target = (SampleRecord("A01", 2, "S01", "env2"),)
    jobs = resolve_report_jobs(tmp_path)

    assert records_for_job(jobs[0], source, target) is source
    assert all(records_for_job(job, source, target) is target for job in jobs[1:])
