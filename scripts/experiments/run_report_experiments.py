"""Run one seed of the delivery experiment suite sequentially."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.split_manifest import load_manifest  # noqa: E402
from experiments.report_suite import (  # noqa: E402
    ExperimentTask,
    FULL_MODEL_ID,
    SuiteConfig,
    build_training_tasks,
    is_task_complete,
)


REGISTRY_FIELDS = (
    "experiment_id",
    "local_id",
    "split_mode",
    "phase",
    "few_shot_key",
    "command",
    "status",
    "started_at",
    "finished_at",
    "duration_seconds",
    "checkpoint_path",
    "manifest_hash",
    "failure",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one seed of the Wi-Pose delivery experiment suite.",
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--split-modes",
        nargs="+",
        choices=("random_frame", "temporal_block"),
        default=("random_frame", "temporal_block"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-visualizations", action="store_true")
    return parser.parse_args(argv)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _registry_row(task: ExperimentTask) -> dict[str, Any]:
    return {
        "experiment_id": task.experiment_id,
        "local_id": task.local_id,
        "split_mode": task.split_mode,
        "phase": task.phase,
        "few_shot_key": task.few_shot_key or "",
        "command": json.dumps(task.command, ensure_ascii=False),
        "status": "planned",
        "started_at": "",
        "finished_at": "",
        "duration_seconds": "",
        "checkpoint_path": str(task.checkpoint_path),
        "manifest_hash": "",
        "failure": "",
    }


def _write_registry(path: Path, rows: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows.values())
    temporary.replace(path)


def _run_command(command: Sequence[str]) -> None:
    print("RUN:", subprocess.list2cmdline(list(command)), flush=True)
    result = subprocess.run(list(command), cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def _reset_partial_output(task: ExperimentTask, output_root: Path) -> None:
    if not task.output_dir.exists():
        return
    resolved_output = task.output_dir.resolve()
    resolved_root = output_root.resolve()
    if resolved_output == resolved_root or resolved_root not in resolved_output.parents:
        raise ValueError(f"Refusing to remove task output outside output root: {resolved_output}")
    shutil.rmtree(resolved_output)


def _write_completion_marker(task: ExperimentTask, manifest_hash: str) -> None:
    marker = task.output_dir / "completed.json"
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "status": "completed",
        "checkpoint_path": str(task.checkpoint_path),
        "manifest_hash": manifest_hash,
        "checkpoint_selection": "minimum_validation_mpjpe",
    }, indent=2), encoding="utf-8")
    temporary.replace(marker)


def _execute_training_task(
    task: ExperimentTask,
    manifest_hash: str,
    output_root: Path,
    resume: bool,
) -> str:
    if resume and is_task_complete(task, manifest_hash):
        return "skipped_complete"
    if task.output_dir.exists():
        if not resume:
            raise FileExistsError(
                f"Task output already exists; use --resume: {task.output_dir}"
            )
        _reset_partial_output(task, output_root)
    task.output_dir.mkdir(parents=True, exist_ok=True)
    (task.output_dir / "command.json").write_text(
        json.dumps({"command": task.command}, indent=2),
        encoding="utf-8",
    )
    _run_command(task.command)
    if not task.checkpoint_path.is_file():
        raise FileNotFoundError(f"Expected checkpoint not produced: {task.checkpoint_path}")
    _write_completion_marker(task, manifest_hash)
    return "trained"


def _evaluation_command(
    config: SuiteConfig,
    task: ExperimentTask,
    manifest_key: str,
    output_dir: Path,
    pose_visualization: bool,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        str(ROOT / "eval.py"),
        "--dataset-root",
        str(config.dataset_root),
        "--checkpoint",
        str(task.checkpoint_path),
        "--split-manifest",
        str(task.manifest_path),
        "--manifest-key",
        manifest_key,
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(config.batch_size),
        "--device",
        config.device,
    ]
    if pose_visualization:
        command.extend([
            "--pose-viz",
            "--pose-viz-sampling",
            "random",
            "--pose-viz-seed",
            str(config.seed),
            "--pose-viz-max-subjects-per-action",
            "2",
            "--pose-viz-dpi",
            "150",
        ])
    return tuple(command)


def _benchmark_command(
    config: SuiteConfig,
    task: ExperimentTask,
    manifest_key: str,
    output_dir: Path,
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(ROOT / "scripts" / "evaluation" / "benchmark_wipose.py"),
        "--dataset-root",
        str(config.dataset_root),
        "--checkpoint",
        str(task.checkpoint_path),
        "--split-manifest",
        str(task.manifest_path),
        "--manifest-key",
        manifest_key,
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(config.batch_size),
        "--device",
        config.device,
    )


def _mechanism_command(
    config: SuiteConfig,
    task: ExperimentTask,
    manifest_key: str,
    output_dir: Path,
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(ROOT / "scripts" / "evaluation" / "export_mechanism_viz.py"),
        "--dataset-root",
        str(config.dataset_root),
        "--checkpoint",
        str(task.checkpoint_path),
        "--split-manifest",
        str(task.manifest_path),
        "--manifest-key",
        manifest_key,
        "--output-dir",
        str(output_dir),
        "--device",
        config.device,
        "--seed",
        str(config.seed),
        "--dpi",
        "150",
    )


def _ensure_manifests(config: SuiteConfig, tasks: Sequence[ExperimentTask]) -> None:
    required = {task.manifest_path for task in tasks}
    if all(path.is_file() and path.with_suffix(".json").is_file() for path in required):
        return
    _run_command((
        sys.executable,
        str(ROOT / "scripts" / "data" / "build_split_manifests.py"),
        "--dataset-root",
        str(config.dataset_root),
        "--output-dir",
        str(config.output_root / "manifests"),
        "--seed",
        str(config.seed),
        "--block-size",
        "16",
    ))


def _postprocess_task(
    config: SuiteConfig,
    task: ExperimentTask,
    *,
    resume: bool,
    skip_visualizations: bool,
) -> None:
    for key in task.evaluation_keys:
        output_dir = task.output_dir / "evaluations" / key
        visualize = not skip_visualizations and key in task.visualization_keys
        summary_path = output_dir / "benchmark_summary.csv"
        pose_dir = output_dir / "pose_viz"
        evaluation_complete = summary_path.is_file() and (
            not visualize or pose_dir.is_dir()
        )
        if not (resume and evaluation_complete):
            _run_command(_evaluation_command(
                config,
                task,
                key,
                output_dir,
                visualize,
            ))
    for key in task.benchmark_keys:
        output_dir = task.output_dir / "benchmark" / key
        if resume and (output_dir / "runtime_metrics.csv").is_file():
            continue
        _run_command(_benchmark_command(config, task, key, output_dir))
    if (
        not skip_visualizations
        and config.seed == 42
        and task.local_id in {FULL_MODEL_ID, "FT8100"}
    ):
        for key in task.visualization_keys:
            output_dir = task.output_dir / "mechanisms" / key
            if resume and (output_dir / "attention_mechanisms.png").is_file():
                continue
            _run_command(_mechanism_command(config, task, key, output_dir))


def _prune_nonselected_checkpoints(task: ExperimentTask) -> None:
    """Keep only the minimum-validation-MPJPE checkpoint after postprocessing."""
    for path in task.output_dir.glob("*.pth"):
        if path.resolve() != task.checkpoint_path.resolve():
            path.unlink()


def _read_one_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, got {len(rows)}")
    return rows[0]


def _write_evaluation_index(
    path: Path,
    tasks: Sequence[ExperimentTask],
    seed: int,
) -> None:
    rows: list[dict[str, str | int]] = []
    metric_names: set[str] = set()
    for task in tasks:
        for key in task.evaluation_keys:
            summary_path = task.output_dir / "evaluations" / key / "benchmark_summary.csv"
            if not summary_path.is_file():
                continue
            metrics = _read_one_csv(summary_path)
            metric_names.update(metrics)
            rows.append({
                "seed": seed,
                "experiment_id": task.experiment_id,
                "local_id": task.local_id,
                "split_mode": task.split_mode,
                "phase": task.phase,
                "manifest_key": key,
                **metrics,
            })
    if not rows:
        return
    leading = (
        "seed",
        "experiment_id",
        "local_id",
        "split_mode",
        "phase",
        "manifest_key",
    )
    fields = [*leading, *sorted(metric_names)]
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    requested_modes = set(args.split_modes)
    ordered_modes = tuple(
        mode for mode in ("random_frame", "temporal_block")
        if mode in requested_modes
    )
    config = SuiteConfig(
        dataset_root=args.dataset_root.resolve(),
        output_root=args.output_root.resolve(),
        split_modes=ordered_modes,
        seed=args.seed,
        source_epochs=args.source_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    tasks = build_training_tasks(config)
    registry_path = config.output_root / "experiment_registry.csv"
    registry = {task.experiment_id: _registry_row(task) for task in tasks}
    _write_registry(registry_path, registry)
    if args.dry_run:
        for task in tasks:
            print(f"{task.experiment_id}: {subprocess.list2cmdline(list(task.command))}")
        return 0

    _ensure_manifests(config, tasks)
    for task in tasks:
        row = registry[task.experiment_id]
        manifest = load_manifest(task.manifest_path, config.dataset_root)
        row["manifest_hash"] = manifest.manifest_hash
        row["started_at"] = _timestamp()
        row["status"] = "running"
        row["failure"] = ""
        _write_registry(registry_path, registry)
        started = time.perf_counter()
        try:
            training_status = _execute_training_task(
                task,
                manifest.manifest_hash,
                config.output_root,
                args.resume,
            )
            _postprocess_task(
                config,
                task,
                resume=args.resume,
                skip_visualizations=args.skip_visualizations,
            )
            _prune_nonselected_checkpoints(task)
            row["status"] = (
                "completed_from_checkpoint"
                if training_status == "skipped_complete" else "completed"
            )
        except KeyboardInterrupt:
            row["status"] = "interrupted"
            row["failure"] = "KeyboardInterrupt"
            raise
        except Exception as error:
            row["status"] = "failed"
            row["failure"] = f"{type(error).__name__}: {error}"
            if not args.continue_on_error:
                row["finished_at"] = _timestamp()
                row["duration_seconds"] = f"{time.perf_counter() - started:.3f}"
                _write_registry(registry_path, registry)
                _write_evaluation_index(
                    config.output_root / "evaluation_index.csv",
                    tasks,
                    config.seed,
                )
                return 1
        finally:
            row["finished_at"] = _timestamp()
            row["duration_seconds"] = f"{time.perf_counter() - started:.3f}"
            _write_registry(registry_path, registry)
            _write_evaluation_index(
                config.output_root / "evaluation_index.csv",
                tasks,
                config.seed,
            )
    return 0 if all(row["status"].startswith("completed") for row in registry.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
