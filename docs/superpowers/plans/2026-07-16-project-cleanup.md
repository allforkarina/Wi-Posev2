# Wi-Posev2 Project Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a compact, handoff-ready main branch while preserving training, evaluation, reproducible report experiments, benchmarks, and demo-video export.

**Architecture:** Keep root entry points stable. Move retained CLIs into four script categories, remove stale diagnostics/tests/presentation code, and retain only v6 experiment evidence.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, SciPy, Matplotlib, scikit-learn, Conda, Git.

---

### Task 1: Record the approved branch baseline

**Files:**
- Modify: Git references only.

- [ ] **Step 1: Verify archive ancestry.**

Run:

```powershell
git merge-base --is-ancestor codex/final-uncleaned-20260716 HEAD
git branch --show-current
```

Expected: exit code 0 and `codex/project-cleanup-20260716`.

- [ ] **Step 2: Commit the approved design and this plan.**

```powershell
git add docs/superpowers/specs/2026-07-16-project-cleanup-design.md docs/superpowers/plans/2026-07-16-project-cleanup.md
git commit -m "Plan project cleanup"
```

### Task 2: Organize retained CLIs by workflow

**Files:**
- Move: `scripts/build_memmap.py` -> `scripts/data/build_memmap.py`
- Move: `scripts/build_groundtruth.py` -> `scripts/data/build_groundtruth.py`
- Move: `scripts/build_split_manifests.py` -> `scripts/data/build_split_manifests.py`
- Move: `scripts/visualize_gt.py` -> `scripts/data/visualize_gt.py`
- Move: `scripts/benchmark_wipose.py` -> `scripts/evaluation/benchmark_wipose.py`
- Move: `scripts/run_report_experiments.py` -> `scripts/experiments/run_report_experiments.py`
- Move: `scripts/export_demo_video.py` -> `scripts/media/export_demo_video.py`
- Modify: `scripts/experiments/run_report_experiments.py`

- [ ] **Step 1: Move the scripts with Git-aware renames.**

```powershell
New-Item -ItemType Directory -Force scripts/data, scripts/evaluation, scripts/experiments, scripts/media
git mv scripts/build_memmap.py scripts/data/build_memmap.py
git mv scripts/build_groundtruth.py scripts/data/build_groundtruth.py
git mv scripts/build_split_manifests.py scripts/data/build_split_manifests.py
git mv scripts/visualize_gt.py scripts/data/visualize_gt.py
git mv scripts/benchmark_wipose.py scripts/evaluation/benchmark_wipose.py
git mv scripts/run_report_experiments.py scripts/experiments/run_report_experiments.py
git mv scripts/export_demo_video.py scripts/media/export_demo_video.py
```

- [ ] **Step 2: Change the runner path expressions exactly.**

```python
str(ROOT / "scripts" / "evaluation" / "benchmark_wipose.py")
str(ROOT / "scripts" / "data" / "build_split_manifests.py")
```

- [ ] **Step 3: Compile and commit the moves.**

```powershell
& 'D:\SoftWare\Anaconda\envs\WiFiPose\python.exe' -m py_compile scripts/data/*.py scripts/evaluation/benchmark_wipose.py scripts/experiments/run_report_experiments.py scripts/media/export_demo_video.py
git add scripts
git commit -m "Organize project scripts by workflow"
```

### Task 3: Remove stale source interfaces

**Files:**
- Modify: `eval.py`, `evaluation/__init__.py`
- Delete: `evaluation/feature_viz.py`, `evaluation/hooks.py`, `pose_targets.py`
- Delete: `scripts/backfill_pck_0_05.py`, `scripts/stem_feature_diagnostic.py`, `scripts/export_report_pose_visualizations.py`

- [ ] **Step 1: Remove feature visualization from `eval.py`.**

Delete feature-viz imports, execution branches, `--feature-viz`, `--num-action-samples`, `--figure-width`, `--figure-height`, and the mutual-exclusion condition. Retain:

```python
parser.add_argument(
    "--pose-viz", action="store_true", default=False,
    help="Generate per-subject joint scatter plots (GT vs Prediction).",
)
```

- [ ] **Step 2: Replace the evaluation package declaration.**

```python
"""Evaluation utilities for WiFlow metrics, benchmarking, and pose visualization."""
```

- [ ] **Step 3: Delete stale modules, search references, and commit.**

```powershell
Remove-Item -LiteralPath evaluation/feature_viz.py,evaluation/hooks.py,pose_targets.py,scripts/backfill_pck_0_05.py,scripts/stem_feature_diagnostic.py,scripts/export_report_pose_visualizations.py -Force
rg -n "feature_viz|evaluation\.hooks|backfill_pck_0_05|stem_feature_diagnostic|export_report_pose_visualizations|pose_targets" train.py eval.py data models evaluation experiments scripts
git add -u
git commit -m "Remove stale diagnostics and report exporters"
```

Expected: the search prints no retained-code references.

### Task 4: Create the handoff surface

**Files:**
- Modify: `scripts/data/build_memmap.py`, `AGENTS.md`, `.gitignore`
- Create: `README.md`, `environment.yml`, `docs/HANDOFF.md`, `docs/EXPERIMENTS.md`, `scripts/README.md`

- [ ] **Step 1: Make source and destination paths mandatory in the memmap builder.**

```python
parser.add_argument("--src", required=True, help="Raw MM-Fi dataset root.")
parser.add_argument("--dst", required=True, help="Destination directory for memmap files.")
```

- [ ] **Step 2: Create `environment.yml`.**

```yaml
name: WiFiPose
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.10
  - pytorch
  - numpy
  - scipy
  - matplotlib
  - pillow
  - scikit-learn
```

- [ ] **Step 3: Document exact workflows and constraints.**

`README.md` must include `build_memmap.py`, manifest build, source training, checkpoint evaluation, and demo-video commands using `<raw-mmfi-root>` and `<memmap-root>` placeholders. `docs/HANDOFF.md` must state CSI memmap `(B, H, W, C)`, model input `(B, C, H, W)`, OpenPose18 `(B, 18, 2)`, manifest hash validation, and checkpoint `train_config` reconstruction. `docs/EXPERIMENTS.md` must cover A1-A4, D1/D3, B1, F1-F5, V2-V4, both splits, the v6 single-seed limitation, and `results/final_report_seed42_v6/`. `scripts/README.md` must describe all seven retained CLIs.

- [ ] **Step 4: Update agent instructions and local-artifact ignores.**

Remove pytest and report-pose export instructions. Add demo-video/FFmpeg and manifest guidance. Add:

```gitignore
.agents/
.claude/
.superpowers/
demo_videos/
demo_videos.zip
final_report_*/
*.zip
```

- [ ] **Step 5: Commit documentation.**

```powershell
git add README.md environment.yml AGENTS.md .gitignore docs/HANDOFF.md docs/EXPERIMENTS.md scripts/README.md scripts/data/build_memmap.py
git commit -m "Document cleaned project handoff"
```

### Task 5: Retain v6 evidence and remove generated artifacts

**Files:**
- Create: `results/final_report_seed42_v6/`
- Delete: `outputs/`, `final_report_seed42_v4_results/`, `final_report_seed42_v6_results/`, `final_report_pose_visualizations_random_frame/`, demo videos, ZIP files.

- [ ] **Step 1: Copy evidence before deletion.**

Keep the registry, four manifests, all `train_log.csv`, all test-set evaluation `benchmark_summary.csv`, `per_action_metrics.csv`, `per_environment_metrics.csv`, `per_joint_metrics.csv`, and all `runtime_metrics.csv`. Preserve their paths relative to `final_report_seed42_v6/`.

```powershell
$source = Resolve-Path 'final_report_seed42_v6_results/outputs/final_report_seed42_v6'
$target = Join-Path (Get-Location) 'results/final_report_seed42_v6'
New-Item -ItemType Directory -Force $target, (Join-Path $target 'manifests')
Copy-Item -LiteralPath (Join-Path $source 'experiment_registry.csv') -Destination $target
Copy-Item -LiteralPath (Join-Path $source 'manifests/*') -Destination (Join-Path $target 'manifests')
Get-ChildItem -LiteralPath $source -Recurse -File | Where-Object {
    $_.Name -eq 'train_log.csv' -or $_.Name -eq 'runtime_metrics.csv' -or
    ($_.FullName -match '\\evaluations\\.+_test\\' -and $_.Name -in @('benchmark_summary.csv','per_action_metrics.csv','per_environment_metrics.csv','per_joint_metrics.csv'))
} | ForEach-Object {
    $relative = $_.FullName.Substring($source.Path.Length + 1)
    $destination = Join-Path $target $relative
    New-Item -ItemType Directory -Force (Split-Path -Parent $destination)
    Copy-Item -LiteralPath $_.FullName -Destination $destination
}
```

- [ ] **Step 2: Verify coverage.**

```powershell
(Import-Csv results/final_report_seed42_v6/experiment_registry.csv).Count
Get-ChildItem results/final_report_seed42_v6/manifests -File | Select-Object Name
```

Expected: 30 rows, two `.npz` manifests, and two `.json` sidecars.

- [ ] **Step 3: Delete generated artifacts only after the coverage check.**

```powershell
Remove-Item -LiteralPath outputs,final_report_seed42_v4_results,final_report_seed42_v6_results,final_report_pose_visualizations_random_frame,demo_videos -Recurse -Force
Remove-Item -LiteralPath demo_videos.zip -Force
Get-ChildItem -Recurse -File -Filter '*.zip' | Remove-Item -Force
git add -f results/final_report_seed42_v6
git commit -m "Retain final v6 experiment evidence"
```

### Task 6: Delete tests and historical project artifacts

**Files:**
- Delete: `tests/`, `docs/superpowers/`, `docs/memmap_migration_plan.md`, `REASONIX.md`, `data/gt_merged/`

- [ ] **Step 1: Remove the approved stale content.**

```powershell
Remove-Item -LiteralPath tests,docs/superpowers,data/gt_merged -Recurse -Force
Remove-Item -LiteralPath docs/memmap_migration_plan.md,REASONIX.md -Force
git add -u
git commit -m "Remove stale tests and historical artifacts"
```

### Task 7: Verify and publish the cleaned main branch

**Files:**
- Modify: Git references only.

- [ ] **Step 1: Compile and import all retained modules.**

```powershell
& 'D:\SoftWare\Anaconda\envs\WiFiPose\python.exe' -m compileall -q train.py eval.py dataloader.py data models evaluation experiments scripts
& 'D:\SoftWare\Anaconda\envs\WiFiPose\python.exe' -c "import train, eval, dataloader, data.memmap_dataset, data.split_manifest, models, evaluation.benchmark, evaluation.pose_viz, experiments.report_suite"
```

- [ ] **Step 2: Run `--help` for root CLIs and all seven moved scripts.**

```powershell
& 'D:\SoftWare\Anaconda\envs\WiFiPose\python.exe' train.py --help
& 'D:\SoftWare\Anaconda\envs\WiFiPose\python.exe' eval.py --help
& 'D:\SoftWare\Anaconda\envs\WiFiPose\python.exe' scripts/data/build_memmap.py --help
& 'D:\SoftWare\Anaconda\envs\WiFiPose\python.exe' scripts/data/build_groundtruth.py --help
& 'D:\SoftWare\Anaconda\envs\WiFiPose\python.exe' scripts/data/build_split_manifests.py --help
& 'D:\SoftWare\Anaconda\envs\WiFiPose\python.exe' scripts/data/visualize_gt.py --help
& 'D:\SoftWare\Anaconda\envs\WiFiPose\python.exe' scripts/evaluation/benchmark_wipose.py --help
& 'D:\SoftWare\Anaconda\envs\WiFiPose\python.exe' scripts/experiments/run_report_experiments.py --help
& 'D:\SoftWare\Anaconda\envs\WiFiPose\python.exe' scripts/media/export_demo_video.py --help
```

- [ ] **Step 3: Check cleanup invariants and tracked evidence.**

```powershell
git status --short
git diff --check codex/final-uncleaned-20260716..HEAD
git ls-files results/final_report_seed42_v6
```

Expected: no unintended files or whitespace errors; selected evidence is tracked.

- [ ] **Step 4: Force-update main after all verification passes.**

```powershell
git push origin HEAD:main --force-with-lease=main:fb79effaeb4ea1ee220763d1a9e74f9be1b614e4
git branch -f main HEAD
git push -u origin codex/project-cleanup-20260716
```

Expected: local and remote main point to the verified cleanup commit; former main-only commits are intentionally excluded.
