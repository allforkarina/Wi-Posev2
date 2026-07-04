from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.pose_viz import save_pose_comparison
import scripts.export_report_pose_visualizations as exporter
from scripts.export_report_pose_visualizations import (
    REPORT_JOBS,
    ResolvedReportJob,
    SampleRecord,
    export_job,
    main,
    parse_args,
    prepare_output_dir,
    records_for_job,
    resolve_report_jobs,
    select_one_sample_per_action,
    write_sample_records,
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


class TinySelectedDataset:
    def __init__(self, records: tuple[SampleRecord, ...]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        return {
            "csi": torch.zeros((64, 3, 114), dtype=torch.float32),
            "kpts18": torch.zeros((18, 2), dtype=torch.float32),
            "meta": {
                "env": record.environment,
                "subject": record.subject,
                "action": record.action,
                "frame_idx": record.dataset_index,
            },
        }


class ZeroPoseModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((inputs.shape[0], 18, 2), device=inputs.device)


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


def test_select_one_sample_per_action_rejects_empty_dataset() -> None:
    dataset = FakeDataset(
        indices=np.asarray([], dtype=np.int64),
        _actions=np.asarray([]),
        _samples=np.asarray([]),
        _envs=np.asarray([]),
    )

    with pytest.raises(ValueError, match="No actions"):
        select_one_sample_per_action(dataset, seed=42)


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


def test_write_sample_records_preserves_absolute_indices(tmp_path: Path) -> None:
    records = (
        SampleRecord("A01", 12, "S01", "env2"),
        SampleRecord("A02", 34, "S02", "env2"),
    )
    path = tmp_path / "indices.csv"

    write_sample_records(path, records)

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {
            "action": "A01",
            "dataset_index": "12",
            "subject": "S01",
            "environment": "env2",
        },
        {
            "action": "A02",
            "dataset_index": "34",
            "subject": "S02",
            "environment": "env2",
        },
    ]


def test_prepare_output_dir_rejects_non_empty_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "visuals"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        prepare_output_dir(output_dir)


def test_export_job_renders_every_selected_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        SampleRecord("A01", 12, "S01", "env2"),
        SampleRecord("A02", 34, "S02", "env2"),
    )
    job = ResolvedReportJob(
        output_name="finetune_540_env2",
        model_label="Finetune 540",
        manifest_key="env2_test",
        checkpoint_path=tmp_path / "checkpoint.pth",
    )
    manifest = SimpleNamespace(
        manifest_hash="manifest-hash",
        source_train_normalization=(0.0, 1.0),
    )
    monkeypatch.setattr(
        exporter,
        "load_checkpoint_model",
        lambda *args, **kwargs: ZeroPoseModel(),
    )
    monkeypatch.setattr(
        exporter,
        "build_selected_dataset",
        lambda *args, **kwargs: TinySelectedDataset(records),
    )

    export_job(
        job=job,
        records=records,
        dataset_root=tmp_path,
        manifest=manifest,
        output_dir=tmp_path / "visuals",
        device=torch.device("cpu"),
        batch_size=2,
        num_workers=0,
    )

    output = tmp_path / "visuals" / "finetune_540_env2"
    assert (output / "A01" / "S01_env2_idx12.png").is_file()
    assert (output / "A02" / "S02_env2_idx34.png").is_file()
    assert not list(output.rglob("*.pdf"))


def test_parse_args_uses_reproducible_defaults(tmp_path: Path) -> None:
    args = parse_args([
        "--dataset-root", str(tmp_path / "dataset"),
        "--experiment-root", str(tmp_path / "experiments"),
        "--output-dir", str(tmp_path / "visuals"),
    ])

    assert args.seed == 42
    assert args.batch_size == 64
    assert args.num_workers == 0
    assert args.device == "cuda"


def test_main_rejects_missing_checkpoint_before_creating_output(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    experiment_root = tmp_path / "experiments"
    dataset_root.mkdir()
    manifest_dir = experiment_root / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "random_frame_seed42.npz").write_bytes(b"placeholder")
    output_dir = tmp_path / "visuals"

    with pytest.raises(FileNotFoundError, match="Required checkpoint"):
        main([
            "--dataset-root", str(dataset_root),
            "--experiment-root", str(experiment_root),
            "--output-dir", str(output_dir),
        ])

    assert not output_dir.exists()
