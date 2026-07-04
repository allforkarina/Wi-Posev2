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
