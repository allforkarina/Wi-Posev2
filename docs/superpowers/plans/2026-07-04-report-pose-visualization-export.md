# Report Pose Visualization Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one Linux command that exports deterministic per-action GT-versus-prediction PNGs for the random-frame source checkpoint and four target-data-scale checkpoints.

**Architecture:** A standalone script resolves the existing report manifest and five checkpoint paths, selects one deterministic absolute test index per action, writes auditable selection CSVs, and performs inference only on those selected samples. The existing pose visualization module exposes a public single-sample renderer so the new exporter reuses the established skeleton style without changing `eval.py` behavior.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, Matplotlib, pathlib, pytest, existing NPY memmap dataset and split-manifest pipeline.

---

## File Map

- Create `scripts/export_report_pose_visualizations.py`: CLI, path resolution, deterministic action sampling, CSV writing, selected-sample inference, and five-job orchestration.
- Modify `evaluation/pose_viz.py`: expose a public PNG renderer with optional dataset index and model label while preserving the existing `run_pose_visualization` behavior.
- Create `tests/test_report_pose_visualization_export.py`: renderer, sampling, path resolution, target-index reuse, output protection, and PNG-only tests.
- Modify `AGENTS.md`: document the script and the one-line Linux command.

### Task 1: Expose the single-sample PNG renderer

**Files:**
- Modify: `evaluation/pose_viz.py:184-270`
- Create: `tests/test_report_pose_visualization_export.py`

- [ ] **Step 1: Write the failing renderer test**

Create `tests/test_report_pose_visualization_export.py` with:

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.pose_viz import save_pose_comparison


def _pose(offset: float = 0.0) -> np.ndarray:
    values = np.linspace(0.1, 0.9, 36, dtype=np.float32).reshape(18, 2)
    return values + offset


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


def test_save_pose_comparison_preserves_legacy_filename_without_index(tmp_path: Path) -> None:
    path = save_pose_comparison(
        target=_pose(),
        prediction=_pose(0.01),
        action="A01",
        subject="S01",
        environment="env1",
        output_dir=tmp_path,
    )

    assert path == tmp_path / "A01" / "S01_env1.png"
```

- [ ] **Step 2: Run the test and verify that the public function is missing**

Run in the established environment:

```powershell
conda activate WiFiPose
pytest tests/test_report_pose_visualization_export.py::test_save_pose_comparison_writes_indexed_png_only -v
```

Expected: collection fails because `save_pose_comparison` is not exported by `evaluation.pose_viz`.

- [ ] **Step 3: Convert the private renderer into a public, backward-compatible renderer**

Replace `_save_individual` in `evaluation/pose_viz.py` with this complete public function:

```python
def save_pose_comparison(
    target: np.ndarray,
    prediction: np.ndarray,
    action: str,
    subject: str,
    environment: str,
    output_dir: Path,
    figure_width: float | None = None,
    figure_height: float | None = None,
    dataset_index: int | None = None,
    model_label: str | None = None,
) -> Path:
    """Save one two-panel GT-versus-prediction pose comparison PNG."""
    if target.shape != (18, 2) or prediction.shape != (18, 2):
        raise ValueError("target and prediction must both have shape [18, 2]")

    fig_w = figure_width or 14.0
    fig_h = figure_height or 6.5
    fig, (ax_scatter, ax_skeleton) = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    x_min, x_max, y_min, y_max = _compute_axes_limits(target, prediction)

    ax_scatter.set_facecolor("#ffffff")
    ax_scatter.grid(True, alpha=0.3, color="#e8e8e8", linewidth=0.5)
    _draw_scatter(ax_scatter, target, prediction)
    ax_scatter.set_xlim(x_min, x_max)
    ax_scatter.set_ylim(y_max, y_min)
    ax_scatter.set_aspect("equal")
    ax_scatter.set_xlabel("Normalized X")
    ax_scatter.set_ylabel("Normalized Y")
    ax_scatter.set_title("Joint Scatter (GT vs Pred)", fontsize=11, fontweight="bold")

    ax_skeleton.set_facecolor("#fafafa")
    ax_skeleton.grid(True, alpha=0.2, color="#d0d0d0", linewidth=0.5)
    _draw_skeleton(
        ax_skeleton,
        target,
        hollow=True,
        bone_linestyle="--",
        bone_color="#aaaaaa",
        base_zorder=1,
    )
    _draw_skeleton(
        ax_skeleton,
        prediction,
        hollow=False,
        bone_linestyle="-",
        bone_color="#333333",
        base_zorder=3,
    )
    ax_skeleton.set_xlim(x_min, x_max)
    ax_skeleton.set_ylim(y_max, y_min)
    ax_skeleton.set_aspect("equal")
    ax_skeleton.set_xlabel("Normalized X")
    ax_skeleton.set_ylabel("Normalized Y")
    ax_skeleton.set_title("Skeleton (GT vs Pred)", fontsize=11, fontweight="bold")
    legend_elements = [
        Line2D([0], [0], color="#aaaaaa", linestyle="--", linewidth=1.2, label="GT"),
        Line2D([0], [0], color="#333333", linestyle="-", linewidth=1.2, label="Prediction"),
    ]
    ax_skeleton.legend(handles=legend_elements, loc="upper right", fontsize=9)

    index_label = f"idx{dataset_index}" if dataset_index is not None else None
    title_parts = [
        part
        for part in (model_label, action, subject, environment, index_label)
        if part
    ]
    fig.suptitle(" / ".join(title_parts), fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.10, wspace=0.25)

    action_dir = output_dir / action
    action_dir.mkdir(parents=True, exist_ok=True)
    safe_subject = subject.replace("/", "_").replace("\\", "_")
    safe_env = environment.replace("/", "_").replace("\\", "_")
    index_suffix = f"_idx{dataset_index}" if dataset_index is not None else ""
    output_path = action_dir / f"{safe_subject}_{safe_env}{index_suffix}.png"
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)
    return output_path
```

Update the existing `run_pose_visualization` call from `_save_individual(...)` to:

```python
save_pose_comparison(
    target=targets_np[i],
    prediction=preds[i],
    action=action,
    subject=subject,
    environment=environment,
    output_dir=viz_dir,
    figure_width=figure_width,
    figure_height=figure_height,
)
```

The optional arguments preserve existing filenames and titles when `eval.py --pose-viz` calls the function.

- [ ] **Step 4: Add and run shape-validation coverage**

Append:

```python
import pytest


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
```

Run:

```powershell
pytest tests/test_report_pose_visualization_export.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit the renderer change**

```bash
git add evaluation/pose_viz.py tests/test_report_pose_visualization_export.py
git commit -m "Expose indexed pose comparison renderer"
```

### Task 2: Implement deterministic per-action selection and report path resolution

**Files:**
- Create: `scripts/export_report_pose_visualizations.py`
- Modify: `tests/test_report_pose_visualization_export.py`

- [ ] **Step 1: Write failing tests for job paths and deterministic selection**

Append:

```python
from dataclasses import dataclass

from scripts.export_report_pose_visualizations import (
    REPORT_JOBS,
    SampleRecord,
    records_for_job,
    resolve_report_jobs,
    select_one_sample_per_action,
)


@dataclass
class FakeDataset:
    indices: np.ndarray
    _actions: np.ndarray
    _samples: np.ndarray
    _envs: np.ndarray


def _fake_dataset(environment: str) -> FakeDataset:
    return FakeDataset(
        indices=np.asarray([2, 4, 6, 8, 10, 12], dtype=np.int64),
        _actions=np.asarray(["unused", "unused", "A01", "unused", "A01", "unused", "A02", "unused", "A02", "unused", "A03", "unused", "A03"]),
        _samples=np.asarray(["X", "X", "S01", "X", "S02", "X", "S01", "X", "S02", "X", "S01", "X", "S02"]),
        _envs=np.asarray(["X", "X", environment, "X", environment, "X", environment, "X", environment, "X", environment, "X", environment]),
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
```

- [ ] **Step 2: Run the tests and verify the script module is missing**

```powershell
pytest tests/test_report_pose_visualization_export.py -v
```

Expected: collection fails because `scripts.export_report_pose_visualizations` does not exist.

- [ ] **Step 3: Add job and sample data structures plus path resolution**

Create `scripts/export_report_pose_visualizations.py` with imports, repository path setup, and:

```python
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
from train import extract_prediction_keypoints, prepare_model_input, select_device  # noqa: E402


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
    ReportJobSpec("source_a1_env1", "Source A1", "env1_test", Path("random_frame/source/a1/best_val_pck_0_2.pth")),
    ReportJobSpec("finetune_540_env2", "Finetune 540", "env2_test", Path("random_frame/finetune_540/f5/best_val_pck_0_2.pth")),
    ReportJobSpec("finetune_810_env2", "Finetune 810", "env2_test", Path("random_frame/finetune_scale/v2/best_val_pck_0_2.pth")),
    ReportJobSpec("finetune_4050_env2", "Finetune 4050", "env2_test", Path("random_frame/finetune_scale/v3/best_val_pck_0_2.pth")),
    ReportJobSpec("finetune_8100_env2", "Finetune 8100", "env2_test", Path("random_frame/finetune_scale/v4/best_val_pck_0_2.pth")),
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
```

- [ ] **Step 4: Add deterministic absolute-index sampling**

Add:

```python
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
```

- [ ] **Step 5: Run selection tests**

```powershell
pytest tests/test_report_pose_visualization_export.py -v
```

Expected: all current tests pass.

- [ ] **Step 6: Commit sampling and job resolution**

```bash
git add scripts/export_report_pose_visualizations.py tests/test_report_pose_visualization_export.py
git commit -m "Add deterministic report pose sampling"
```

### Task 3: Implement CSV output, inference, and five-checkpoint orchestration

**Files:**
- Modify: `scripts/export_report_pose_visualizations.py`
- Modify: `tests/test_report_pose_visualization_export.py`

- [ ] **Step 1: Write failing tests for CSV output and non-empty output protection**

Append:

```python
import csv

from scripts.export_report_pose_visualizations import prepare_output_dir, write_sample_records


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
        {"action": "A01", "dataset_index": "12", "subject": "S01", "environment": "env2"},
        {"action": "A02", "dataset_index": "34", "subject": "S02", "environment": "env2"},
    ]


def test_prepare_output_dir_rejects_non_empty_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "visuals"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        prepare_output_dir(output_dir)
```

- [ ] **Step 2: Run tests and verify the helper functions are missing**

```powershell
pytest tests/test_report_pose_visualization_export.py -v
```

Expected: collection fails on missing `prepare_output_dir` and `write_sample_records`.

- [ ] **Step 3: Implement output protection and CSV writing**

Add:

```python
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
```

- [ ] **Step 4: Implement selected-sample dataset creation and inference**

Add:

```python
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
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    rendered = 0
    with torch.no_grad():
        for batch in loader:
            model_input, target = prepare_model_input(batch, device)
            prediction = extract_prediction_keypoints(model(model_input)).cpu().numpy()
            target_np = target.cpu().numpy()
            for offset in range(len(prediction)):
                save_pose_comparison(
                    target=target_np[offset],
                    prediction=prediction[offset],
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
```

- [ ] **Step 5: Implement CLI parsing, path validation, and orchestration**

Add:

```python
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
    manifest_path = experiment_root / "manifests" / f"random_frame_seed{args.seed}.npz"
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Random-frame manifest not found: {manifest_path}")

    jobs = resolve_report_jobs(experiment_root)
    missing = [job.checkpoint_path for job in jobs if not job.checkpoint_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required checkpoint not found: {missing[0]}")
    prepare_output_dir(output_dir)

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
    write_sample_records(output_dir / "sample_indices" / f"env1_test_seed{args.seed}.csv", source_records)
    write_sample_records(output_dir / "sample_indices" / f"env2_test_seed{args.seed}.csv", target_records)

    device = select_device(args.device)
    for job in jobs:
        records = records_for_job(job, source_records, target_records)
        export_job(
            job,
            records,
            dataset_root,
            manifest,
            output_dir,
            device,
            args.batch_size,
            args.num_workers,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run the focused test file**

```powershell
pytest tests/test_report_pose_visualization_export.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit the exporter pipeline**

```bash
git add scripts/export_report_pose_visualizations.py tests/test_report_pose_visualization_export.py
git commit -m "Export report pose comparisons"
```

### Task 4: Document and verify the completed workflow

**Files:**
- Modify: `AGENTS.md`
- Verify: `scripts/export_report_pose_visualizations.py`
- Verify: `evaluation/pose_viz.py`
- Verify: `tests/test_report_pose_visualization_export.py`

- [ ] **Step 1: Add the script to the repository structure documentation**

Add under the `scripts/` list in `AGENTS.md`:

```markdown
- `scripts/export_report_pose_visualizations.py`: Exports deterministic random-frame final-report pose PNGs for Source A1 and the 540/810/4050/8100-frame full-finetune checkpoints. It selects one seeded random test frame per action, reuses identical env2 indices across all finetuning scales, and records the selected absolute indices in CSV files.
```

- [ ] **Step 2: Add the one-line Linux command**

Add under evaluation commands in `AGENTS.md`:

```bash
python scripts/export_report_pose_visualizations.py --dataset-root /data/WiFiPose/dataset/mmfi_pose_v4 --experiment-root outputs/final_report_seed42_v4 --output-dir outputs/final_report_seed42_v4/pose_visualizations/random_frame --seed 42 --batch-size 64 --num-workers 4 --device cuda
```

Document that the command requires the five `best_val_pck_0_2.pth` files to remain inside their original experiment directories.

- [ ] **Step 3: Run focused and regression tests**

```powershell
conda activate WiFiPose
pytest tests/test_report_pose_visualization_export.py tests/test_eval_png_only.py tests/test_manifest_pipeline.py -v
```

Expected: all selected tests pass.

- [ ] **Step 4: Run the full test suite**

```powershell
pytest
```

Expected: all tests pass. If CUDA is unavailable locally, the tests still use synthetic CPU fixtures and must not require a real checkpoint.

- [ ] **Step 5: Run static command validation**

```powershell
python scripts/export_report_pose_visualizations.py --help
```

Expected: help lists `--dataset-root`, `--experiment-root`, `--output-dir`, `--seed`, `--batch-size`, `--num-workers`, and `--device`.

- [ ] **Step 6: Inspect the final diff**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the exporter, renderer, tests, plan-related documentation, and `AGENTS.md` are modified. Existing generated results and unrelated untracked directories remain untouched.

- [ ] **Step 7: Commit and push the implementation**

```bash
git add AGENTS.md evaluation/pose_viz.py scripts/export_report_pose_visualizations.py tests/test_report_pose_visualization_export.py
git commit -m "Add final report pose visualization export"
git push origin codex/release2-physical-csi
```

- [ ] **Step 8: Verify Linux output after the server run**

After pulling the branch and running the documented command on the Linux server, verify:

```bash
find outputs/final_report_seed42_v4/pose_visualizations/random_frame -type f -name '*.png' | wc -l
find outputs/final_report_seed42_v4/pose_visualizations/random_frame -type f -name '*.pdf' | wc -l
```

Expected: PNG count equals five times the number of actions represented across the source/target selections when both environments contain the same action set; PDF count is `0`. Confirm the exact counts per model with:

```bash
for d in source_a1_env1 finetune_540_env2 finetune_810_env2 finetune_4050_env2 finetune_8100_env2; do printf '%s ' "$d"; find "outputs/final_report_seed42_v4/pose_visualizations/random_frame/$d" -type f -name '*.png' | wc -l; done
```
