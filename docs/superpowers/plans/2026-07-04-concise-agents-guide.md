# Concise AGENTS.md Contributor Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a concise 200–400-word root `AGENTS.md` contributor guide for Wi-Posev2.

**Architecture:** Replace the currently missing root guide with one self-contained Markdown document. Keep repository structure, commands, style, testing, Git expectations, security, and minimal agent-specific rules actionable without restoring the removed long-form domain reference.

**Tech Stack:** Markdown, Python 3.10+, Conda, pytest, Git.

---

### Task 1: Create and verify the contributor guide

**Files:**
- Create: `AGENTS.md`
- Verify: `train.py`, `eval.py`, `scripts/export_report_pose_visualizations.py`, `tests/`

- [ ] **Step 1: Create `AGENTS.md` with the approved content**

```markdown
# Repository Guidelines

## Project Structure & Module Organization

Core entry points are `train.py` for source training and cross-domain finetuning and `eval.py` for metrics and PNG visualizations. Model components live in `models/`; memmap loading and deterministic manifests live in `data/` and `dataloader.py`. Evaluation helpers are under `evaluation/`, report-suite definitions under `experiments/`, and preprocessing, benchmarking, and export commands under `scripts/`. Tests mirror modules in `tests/`. Keep generated datasets, checkpoints, and `outputs/` outside Git.

## Build, Test, and Development Commands

Use the existing environment:

```bash
conda activate WiFiPose
pytest
```

Run a small training check with `python train.py --mode source_only --dataset-root data/mmfi_pose --epochs 1 --subset-size 32 --output-dir outputs/sanity`. Evaluate a checkpoint with `python eval.py --dataset-root data/mmfi_pose --checkpoint outputs/train/best_val_pck_0_2.pth --output-dir outputs/eval`. Export final-report pose comparisons with `python scripts/export_report_pose_visualizations.py --help`.

## Coding Style & Naming Conventions

Target Python 3.10+, use four-space indentation, type hints, and `pathlib.Path`. Follow `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for constants. Group imports as standard library, third-party, then local modules. Keep comments focused on CSI tensor shapes, normalization, and physical assumptions. Avoid unrelated refactors.

## Testing Guidelines

Use pytest. Name files `test_*.py` and tests `test_<behavior>()`. Add focused coverage for shape contracts, manifest indices, path validation, checkpoint reconstruction, and PNG-only output. Prefer tiny synthetic fixtures and temporary directories; never require the full dataset in unit tests. Run focused tests during development and `pytest` before completion.

## Commit & Pull Request Guidelines

Recent history uses short imperative commits such as `Export report pose comparisons`. Keep commits scoped. Pull requests should summarize behavior, list commands run, state dataset and split assumptions, and include representative PNGs when visualization output changes. Do not include generated data or checkpoints.

## Security & Agent-Specific Instructions

Pass dataset locations through CLI arguments; never hard-code private server paths. Preserve unrelated user changes. Write repository-facing code and documentation in English and communicate with the user in Chinese unless requested otherwise. Use the `WiFiPose` environment, verify changes before claiming completion, and after project modifications commit and push the active `codex/` branch unless explicitly told not to.
```

- [ ] **Step 2: Verify title, word count, paths, and Markdown fences**

Run:

```powershell
$text=Get-Content -LiteralPath AGENTS.md -Raw; (($text -split '\s+') | Where-Object {$_}).Count; Get-Content AGENTS.md -TotalCount 1; Test-Path train.py,eval.py,scripts\export_report_pose_visualizations.py,tests
```

Expected: word count is between 200 and 400; first line is `# Repository Guidelines`; all paths return `True`.

- [ ] **Step 3: Check formatting and isolate the intended change**

```bash
git diff --check -- AGENTS.md
git status --short
```

Expected: no whitespace errors. `AGENTS.md` is the only file created by this implementation; pre-existing deletions and untracked result directories remain untouched.

- [ ] **Step 4: Commit and push**

```bash
git add AGENTS.md
git commit -m "Add concise repository guidelines"
git push origin codex/release2-physical-csi
```
