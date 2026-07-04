from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.memmap_dataset import MemmapDataset  # noqa: E402
from data.split_manifest import SplitManifest, load_manifest  # noqa: E402
from dataloader import memmap_collate_fn  # noqa: E402
from eval import load_checkpoint_model  # noqa: E402
from evaluation.pose_viz import save_pose_comparison  # noqa: E402
from train import (  # noqa: E402
    extract_prediction_keypoints,
    prepare_model_input,
    select_device,
)


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


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def write_sample_records(path: Path, records: Sequence[SampleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("action", "dataset_index", "subject", "environment"),
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "action": record.action,
                "dataset_index": record.dataset_index,
                "subject": record.subject,
                "environment": record.environment,
            })


def build_selected_dataset(
    dataset_root: Path,
    manifest: SplitManifest,
    records: Sequence[SampleRecord],
) -> MemmapDataset:
    return MemmapDataset(
        data_dir=dataset_root,
        split="all",
        indices=[record.dataset_index for record in records],
        split_normalization=manifest.source_train_normalization,
    )


def export_job(
    job: ResolvedReportJob,
    records: Sequence[SampleRecord],
    dataset_root: Path,
    manifest: SplitManifest,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> None:
    model = load_checkpoint_model(
        job.checkpoint_path,
        device,
        expected_manifest_hash=manifest.manifest_hash,
    )
    dataset = build_selected_dataset(dataset_root, manifest, records)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=memmap_collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    rendered = 0
    with torch.no_grad():
        for batch in loader:
            model_input, target = prepare_model_input(batch, device)
            predictions = extract_prediction_keypoints(model(model_input)).cpu().numpy()
            targets = target.cpu().numpy()
            for offset in range(len(predictions)):
                save_pose_comparison(
                    target=targets[offset],
                    prediction=predictions[offset],
                    action=str(batch["action"][offset]),
                    subject=str(batch["sample"][offset]),
                    environment=str(batch["environment"][offset]),
                    output_dir=output_dir / job.output_name,
                    dataset_index=int(batch["frame_idx"][offset]),
                    model_label=job.model_label,
                )
                rendered += 1
    if rendered != len(records):
        raise RuntimeError(f"Rendered {rendered} samples, expected {len(records)}")


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export final-report GT-versus-prediction pose PNGs.",
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    experiment_root = args.experiment_root.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = (
        experiment_root / "manifests" / f"random_frame_seed{args.seed}.npz"
    )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Random-frame manifest not found: {manifest_path}")

    jobs = resolve_report_jobs(experiment_root)
    missing = [job.checkpoint_path for job in jobs if not job.checkpoint_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required checkpoint not found: {missing[0]}")

    manifest = load_manifest(manifest_path, dataset_root)
    source_candidates = MemmapDataset(
        dataset_root,
        split="all",
        indices=manifest.indices("env1_test"),
        split_normalization=manifest.source_train_normalization,
    )
    target_candidates = MemmapDataset(
        dataset_root,
        split="all",
        indices=manifest.indices("env2_test"),
        split_normalization=manifest.source_train_normalization,
    )
    source_records = select_one_sample_per_action(source_candidates, args.seed)
    target_records = select_one_sample_per_action(target_candidates, args.seed)

    prepare_output_dir(output_dir)
    sample_dir = output_dir / "sample_indices"
    write_sample_records(
        sample_dir / f"env1_test_seed{args.seed}.csv",
        source_records,
    )
    write_sample_records(
        sample_dir / f"env2_test_seed{args.seed}.csv",
        target_records,
    )

    device = select_device(args.device)
    for job in jobs:
        records = records_for_job(job, source_records, target_records)
        print(f"Exporting {job.output_name}: {len(records)} actions", flush=True)
        export_job(
            job=job,
            records=records,
            dataset_root=dataset_root,
            manifest=manifest,
            output_dir=output_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
