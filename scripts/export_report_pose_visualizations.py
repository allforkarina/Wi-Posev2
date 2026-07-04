from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.memmap_dataset import MemmapDataset  # noqa: E402


@dataclass(frozen=True)
class ReportJobSpec:
    output_name: str
    model_label: str
    manifest_key: str
    checkpoint_relative_path: Path


@dataclass(frozen=True)
class ResolvedReportJob:
    output_name: str
    model_label: str
    manifest_key: str
    checkpoint_path: Path


@dataclass(frozen=True)
class SampleRecord:
    action: str
    dataset_index: int
    subject: str
    environment: str


REPORT_JOBS = (
    ReportJobSpec(
        "source_a1_env1",
        "Source A1",
        "env1_test",
        Path("random_frame/source/a1/best_val_pck_0_2.pth"),
    ),
    ReportJobSpec(
        "finetune_540_env2",
        "Finetune 540",
        "env2_test",
        Path("random_frame/finetune_540/f5/best_val_pck_0_2.pth"),
    ),
    ReportJobSpec(
        "finetune_810_env2",
        "Finetune 810",
        "env2_test",
        Path("random_frame/finetune_scale/v2/best_val_pck_0_2.pth"),
    ),
    ReportJobSpec(
        "finetune_4050_env2",
        "Finetune 4050",
        "env2_test",
        Path("random_frame/finetune_scale/v3/best_val_pck_0_2.pth"),
    ),
    ReportJobSpec(
        "finetune_8100_env2",
        "Finetune 8100",
        "env2_test",
        Path("random_frame/finetune_scale/v4/best_val_pck_0_2.pth"),
    ),
)


def resolve_report_jobs(experiment_root: Path) -> tuple[ResolvedReportJob, ...]:
    return tuple(
        ResolvedReportJob(
            output_name=spec.output_name,
            model_label=spec.model_label,
            manifest_key=spec.manifest_key,
            checkpoint_path=experiment_root / spec.checkpoint_relative_path,
        )
        for spec in REPORT_JOBS
    )


def records_for_job(
    job: ResolvedReportJob,
    source_records: tuple[SampleRecord, ...],
    target_records: tuple[SampleRecord, ...],
) -> tuple[SampleRecord, ...]:
    if job.manifest_key == "env1_test":
        return source_records
    if job.manifest_key == "env2_test":
        return target_records
    raise ValueError(f"Unsupported report manifest key: {job.manifest_key}")


def select_one_sample_per_action(
    dataset: MemmapDataset,
    seed: int,
) -> tuple[SampleRecord, ...]:
    grouped: dict[str, list[int]] = {}
    for absolute_index in dataset.indices:
        index = int(absolute_index)
        action = str(dataset._actions[index])
        grouped.setdefault(action, []).append(index)
    if not grouped:
        raise ValueError("No actions are available for pose visualization")

    rng = np.random.default_rng(seed)
    selected: list[SampleRecord] = []
    for action in sorted(grouped):
        candidates = np.asarray(sorted(grouped[action]), dtype=np.int64)
        dataset_index = int(rng.choice(candidates))
        selected.append(SampleRecord(
            action=action,
            dataset_index=dataset_index,
            subject=str(dataset._samples[dataset_index]),
            environment=str(dataset._envs[dataset_index]),
        ))
    return tuple(selected)
