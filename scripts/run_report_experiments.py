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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.split_manifest import load_manifest  # noqa: E402
from experiments.report_suite import (  # noqa: E402
    ExperimentTask,
    SuiteConfig,
    build_training_tasks,
    is_task_complete,
    resolve_scale_task,
    select_trainable_group,
)


REGISTRY_FIELDS = (
    "experiment_id",
    "split_mode",
    "phase",
    "command",
    "status",
    "started_at",
    "finished_at",
    "duration_seconds",
    "checkpoint_path",
    "manifest_hash",
    "val_pck_0_2",
    "val_mpjpe",
    "test_pck_0_2",
    "test_mpjpe",
    "failure",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the single-seed Wi-Pose final-report experiment suite.",
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
    return parser.parse_args(argv)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _registry_row(task: ExperimentTask, status: str = "planned") -> dict[str, Any]:
    return {
        "experiment_id": task.experiment_id,
        "split_mode": task.split_mode,
        "phase": task.phase,
        "command": json.dumps(task.command, ensure_ascii=False),
        "status": status,
        "started_at": "",
        "finished_at": "",
        "duration_seconds": "",
        "checkpoint_path": str(task.checkpoint_path),
        "manifest_hash": "",
        "val_pck_0_2": "",
        "val_mpjpe": "",
        "test_pck_0_2": "",
        "test_mpjpe": "",
        "failure": "",
    }


def _write_registry(path: Path, rows: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)
    temporary.replace(path)


def _read_summary(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation summary not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one summary row in {path}, got {len(rows)}")
    return rows[0]


def _run_command(command: Sequence[str]) -> None:
    print("RUN:", subprocess.list2cmdline(list(command)), flush=True)
    result = subprocess.run(list(command), cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def _run_postprocess(
    task: ExperimentTask,
    command: Sequence[str],
    stage: str,
    registry_path: Path,
    registry: dict[str, dict[str, Any]],
) -> bool:
    try:
        _run_command(command)
        return True
    except Exception as error:
        row = registry[task.experiment_id]
        row["status"] = "failed"
        row["failure"] = f"{stage}: {type(error).__name__}: {error}"
        _write_registry(registry_path, registry)
        return False


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
    }, indent=2), encoding="utf-8")
    temporary.replace(marker)


def _execute_training_task(
    task: ExperimentTask,
    manifest_hash: str,
    output_root: Path,
    registry_path: Path,
    registry: dict[str, dict[str, Any]],
    resume: bool,
) -> bool:
    row = registry[task.experiment_id]
    if resume and is_task_complete(task, manifest_hash):
        row["status"] = "skipped_complete"
        row["manifest_hash"] = manifest_hash
        _write_registry(registry_path, registry)
        return True
    if task.output_dir.exists():
        if not resume:
            raise FileExistsError(
                f"Task output already exists; use --resume to validate/rerun it: {task.output_dir}"
            )
        _reset_partial_output(task, output_root)
    task.output_dir.mkdir(parents=True, exist_ok=True)
    (task.output_dir / "command.json").write_text(
        json.dumps({"command": task.command}, indent=2),
        encoding="utf-8",
    )
    row.update({
        "command": json.dumps(task.command, ensure_ascii=False),
        "status": "running",
        "started_at": _timestamp(),
        "manifest_hash": manifest_hash,
        "failure": "",
    })
    _write_registry(registry_path, registry)
    start = time.perf_counter()
    try:
        _run_command(task.command)
        if not task.checkpoint_path.is_file():
            raise FileNotFoundError(f"Expected checkpoint not produced: {task.checkpoint_path}")
        _write_completion_marker(task, manifest_hash)
        row["status"] = "completed"
        return True
    except Exception as error:
        row["status"] = "failed"
        row["failure"] = f"{type(error).__name__}: {error}"
        return False
    finally:
        row["finished_at"] = _timestamp()
        row["duration_seconds"] = f"{time.perf_counter() - start:.3f}"
        _write_registry(registry_path, registry)


def _evaluation_command(
    config: SuiteConfig,
    task: ExperimentTask,
    manifest_key: str,
    output_dir: Path,
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(ROOT / "eval.py"),
        "--dataset-root", str(config.dataset_root),
        "--checkpoint", str(task.checkpoint_path),
        "--split-manifest", str(task.manifest_path),
        "--manifest-key", manifest_key,
        "--output-dir", str(output_dir),
        "--batch-size", str(config.batch_size),
        "--device", config.device,
    )


def _benchmark_command(
    config: SuiteConfig,
    task: ExperimentTask,
    manifest_key: str,
    output_dir: Path,
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(ROOT / "scripts" / "benchmark_wipose.py"),
        "--dataset-root", str(config.dataset_root),
        "--checkpoint", str(task.checkpoint_path),
        "--split-manifest", str(task.manifest_path),
        "--manifest-key", manifest_key,
        "--output-dir", str(output_dir),
        "--batch-size", str(config.batch_size),
        "--device", config.device,
    )


def _ensure_manifests(config: SuiteConfig, tasks: Sequence[ExperimentTask]) -> None:
    required = {task.manifest_path for task in tasks}
    if all(path.is_file() and path.with_suffix(".json").is_file() for path in required):
        return
    _run_command((
        sys.executable,
        str(ROOT / "scripts" / "build_split_manifests.py"),
        "--dataset-root", str(config.dataset_root),
        "--output-dir", str(config.output_root / "manifests"),
        "--seed", str(config.seed),
        "--block-size", "16",
    ))


def _record_summary(
    registry: dict[str, dict[str, Any]],
    task: ExperimentTask,
    summary: Mapping[str, str],
    prefix: str,
) -> None:
    registry[task.experiment_id][f"{prefix}_pck_0_2"] = summary["pck_0_2"]
    registry[task.experiment_id][f"{prefix}_mpjpe"] = summary["mpjpe"]


def _record_source_summary(
    registry: dict[str, dict[str, Any]],
    task: ExperimentTask,
    manifest_key: str,
    summary: Mapping[str, str],
) -> None:
    if manifest_key == "env1_val":
        _record_summary(registry, task, summary, "val")
    elif manifest_key == "env1_test":
        _record_summary(registry, task, summary, "test")


def _run_split(
    config: SuiteConfig,
    split_tasks: list[ExperimentTask],
    registry_path: Path,
    registry: dict[str, dict[str, Any]],
    resume: bool,
    continue_on_error: bool,
) -> bool:
    manifest = load_manifest(split_tasks[0].manifest_path, config.dataset_root)
    source_tasks = [task for task in split_tasks if task.phase == "source"]
    layer_tasks = [task for task in split_tasks if task.phase == "finetune_540"]
    scale_tasks = [task for task in split_tasks if task.phase == "finetune_scale"]

    for task in source_tasks:
        succeeded = _execute_training_task(
            task, manifest.manifest_hash, config.output_root, registry_path, registry, resume
        )
        if not succeeded:
            if not continue_on_error:
                return False
            continue
        postprocess_ok = True
        for key in ("env1_val", "env1_test", "env2_val", "env2_test"):
            evaluation_dir = task.output_dir / "evaluations" / key
            if not _run_postprocess(
                task,
                _evaluation_command(config, task, key, evaluation_dir),
                f"evaluation:{key}",
                registry_path,
                registry,
            ):
                postprocess_ok = False
                break
            _record_source_summary(
                registry,
                task,
                key,
                _read_summary(evaluation_dir / "benchmark_summary.csv"),
            )
            _write_registry(registry_path, registry)
        if postprocess_ok:
            postprocess_ok = _run_postprocess(
                task,
                _benchmark_command(
                    config, task, "env1_test", task.output_dir / "benchmark" / "env1_test"
                ),
                "benchmark:env1_test",
                registry_path,
                registry,
            )
        if not postprocess_ok and not continue_on_error:
            return False

    validation_rows: list[dict[str, str]] = []
    successful_layer_tasks: list[ExperimentTask] = []
    for task in layer_tasks:
        succeeded = _execute_training_task(
            task, manifest.manifest_hash, config.output_root, registry_path, registry, resume
        )
        if not succeeded:
            if not continue_on_error:
                return False
            continue
        validation_dir = task.output_dir / "evaluations" / "env2_val"
        if not _run_postprocess(
            task,
            _evaluation_command(config, task, "env2_val", validation_dir),
            "evaluation:env2_val",
            registry_path,
            registry,
        ):
            if not continue_on_error:
                return False
            continue
        summary = _read_summary(validation_dir / "benchmark_summary.csv")
        _record_summary(registry, task, summary, "val")
        validation_rows.append({
            "experiment_id": task.experiment_id,
            "trainable_group": str(task.trainable_group),
            "pck_0_2": summary["pck_0_2"],
            "mpjpe": summary["mpjpe"],
        })
        successful_layer_tasks.append(task)
        _write_registry(registry_path, registry)
    if len(validation_rows) != 5:
        return False
    selected_group = select_trainable_group(validation_rows)

    resolved_scale_tasks = [resolve_scale_task(task, selected_group) for task in scale_tasks]
    successful_finetunes = list(successful_layer_tasks)
    for task in resolved_scale_tasks:
        registry[task.experiment_id]["command"] = json.dumps(task.command, ensure_ascii=False)
        succeeded = _execute_training_task(
            task, manifest.manifest_hash, config.output_root, registry_path, registry, resume
        )
        if not succeeded:
            if not continue_on_error:
                return False
            continue
        successful_finetunes.append(task)

    for task in successful_finetunes:
        test_dir = task.output_dir / "evaluations" / "env2_test"
        if not _run_postprocess(
            task,
            _evaluation_command(config, task, "env2_test", test_dir),
            "evaluation:env2_test",
            registry_path,
            registry,
        ):
            if not continue_on_error:
                return False
            continue
        summary = _read_summary(test_dir / "benchmark_summary.csv")
        _record_summary(registry, task, summary, "test")
        if not _run_postprocess(
            task,
            _benchmark_command(
                config, task, "env2_test", task.output_dir / "benchmark" / "env2_test"
            ),
            "benchmark:env2_test",
            registry_path,
            registry,
        ) and not continue_on_error:
            return False
        _write_registry(registry_path, registry)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    requested_modes = set(args.split_modes)
    ordered_modes = tuple(
        mode
        for mode in ("random_frame", "temporal_block")
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
    if args.dry_run:
        _write_registry(registry_path, registry)
        for task in tasks:
            print(f"{task.experiment_id}: {subprocess.list2cmdline(list(task.command))}")
        return 0

    _ensure_manifests(config, tasks)
    for split_mode in config.split_modes:
        split_tasks = [task for task in tasks if task.split_mode == split_mode]
        succeeded = _run_split(
            config,
            split_tasks,
            registry_path,
            registry,
            resume=args.resume,
            continue_on_error=args.continue_on_error,
        )
        if not succeeded and not args.continue_on_error:
            return 1
    return 0 if all(row["status"] != "failed" for row in registry.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
