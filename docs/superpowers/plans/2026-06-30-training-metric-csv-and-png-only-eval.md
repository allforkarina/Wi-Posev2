# Training Metric CSV and PNG-Only Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist each epoch's loss, MPJPE, threshold-specific PCK, and bone error in separate CSV files, and make evaluation visualizations PNG-only.

**Architecture:** Extend the existing batch metric aggregation with a topology-aware `bone_error`, then route completed epoch dictionaries through one focused CSV writer. Keep `train_log.csv` and evaluation metric CSVs unchanged. Simplify the feature visualization save path to one unconditional PNG write.

**Tech Stack:** Python 3.10+, PyTorch, Matplotlib, pytest, pathlib/csv.

---

### Task 1: Per-epoch training metric artifacts

**Files:**
- Create: `tests/test_training_metric_artifacts.py`
- Modify: `train.py`

- [ ] **Step 1: Write failing tests for metric computation and CSV separation**

Add tests that construct known `[B, 18, 2]` tensors and assert
`compute_metrics(...)` contains `bone_error`; use `tmp_path` to call the desired
`append_epoch_metric_csvs(...)` API twice and assert that `loss.csv`,
`mpjpe.csv`, `bone_error.csv`, and all five `pck_0_x.csv` files contain a header
plus two epoch rows. Assert optional loss keys stay in `loss.csv` and do not
appear in performance files.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
conda activate WiFiPose
pytest tests/test_training_metric_artifacts.py -v
```

Expected: collection/import failure because `append_epoch_metric_csvs` and the
`bone_error` metric do not exist.

- [ ] **Step 3: Implement the smallest metric and writer changes**

In `compute_metrics`, add:

```python
metrics = {
    "mpjpe": mpjpe(prediction, target),
    "bone_error": bone_length_loss(prediction, target),
}
```

Add `append_epoch_metric_csvs(output_dir, epoch, train_metrics, val_metrics=None)`
beside `append_csv_row`. It writes:

```python
loss_names = (
    "loss", "coord_loss", "bone_loss", "latent_structure_loss",
    "encoder_relation_loss", "distal_loss", "wrist_direction_loss",
    "target_loss", "source_loss",
)
```

Use the existing `append_csv_row` for `loss.csv`, `mpjpe.csv`,
`bone_error.csv`, and one file for every name in `PCK_THRESHOLDS`. Rows always
start with `epoch`, include `train_*`, and include `val_*` only when a validation
dictionary is supplied.

Merge `compute_metrics(prediction.detach(), target)` into
`run_finetune_epoch` totals so finetuning exposes the same target-domain
performance metrics as source-only training. Call the writer once per completed
epoch from `_run_source_only` and `_run_finetune` after the existing
`train_log.csv` append.

- [ ] **Step 4: Run focused tests and verify GREEN**

```powershell
conda activate WiFiPose
pytest tests/test_training_metric_artifacts.py tests/test_distal_replay_training.py -v
```

Expected: all selected tests pass.

### Task 2: PNG-only feature visualization

**Files:**
- Create: `tests/test_eval_png_only.py`
- Modify: `eval.py`
- Modify: `evaluation/feature_viz.py`

- [ ] **Step 1: Write failing PNG-only tests**

Test `_save_fig` with a temporary base path and assert `.png` exists while
`.pdf` does not. Patch `sys.argv` with the required `eval.py` arguments and
assert `parse_args()` has no `output_format` attribute. Add a subprocess-style
parser test asserting legacy `--output-format pdf` is rejected.

- [ ] **Step 2: Run the tests and verify RED**

```powershell
conda activate WiFiPose
pytest tests/test_eval_png_only.py -v
```

Expected: PDF is created by the current default and `parse_args()` still exposes
`output_format`.

- [ ] **Step 3: Remove format branching**

Delete `_OUTPUT_FORMAT`, make `_save_fig` unconditionally call:

```python
fig.savefig(str(path.with_suffix(".png")), dpi=300)
```

Remove `output_format` from `run_feature_visualization`, its docstring, global
assignment, the `eval.py` parser, and the call from `main()`.

- [ ] **Step 4: Run focused tests and verify GREEN**

```powershell
conda activate WiFiPose
pytest tests/test_eval_png_only.py tests/test_feature_viz_layout.py -v
```

Expected: all selected tests pass and no PDF artifact is created.

### Task 3: Repository documentation and full verification

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Document the changed workflow**

Update the `train.py`, `eval.py`, and output-artifact descriptions to name
`loss.csv`, `mpjpe.csv`, `pck_0_1.csv` through `pck_0_5.csv`, and
`bone_error.csv`; state that evaluation visualizations are PNG-only.

- [ ] **Step 2: Run complete verification**

```powershell
conda activate WiFiPose
pytest
```

Expected: complete suite passes with zero failures.

- [ ] **Step 3: Inspect the final patch**

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; only the plan, tests, `train.py`, `eval.py`,
`evaluation/feature_viz.py`, and `AGENTS.md` are changed, excluding pre-existing
untracked user directories.

- [ ] **Step 4: Commit and push**

```powershell
git add AGENTS.md train.py eval.py evaluation/feature_viz.py tests/test_training_metric_artifacts.py tests/test_eval_png_only.py docs/superpowers/plans/2026-06-30-training-metric-csv-and-png-only-eval.md
git commit -m "Add per-epoch training metric artifacts"
git push
```
