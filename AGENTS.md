# Repository Guidelines

## Project Structure & Module Organization

Core entry points are `train.py` for source training and cross-domain finetuning and `eval.py` for metrics and optional pose visualizations. Model components live in `models/`; memmap loading and deterministic manifests live in `data/` and `dataloader.py`. Evaluation helpers are under `evaluation/`, report-suite definitions under `experiments/`, and categorized standalone commands live under `scripts/`. Keep generated datasets, checkpoints, videos, reports, and `outputs/` outside Git.

## Build, Test, and Development Commands

Use the existing environment:

```bash
conda activate WiFiPose
```

Build manifests with `python scripts/data/build_split_manifests.py --dataset-root <memmap-root> --output-dir outputs/manifests`. Run a small training check with `python train.py --mode source_only --dataset-root <memmap-root> --epochs 1 --subset-size 32 --output-dir outputs/sanity`. Evaluate a checkpoint with `python eval.py --dataset-root <memmap-root> --checkpoint outputs/train/best_val_pck_0_2.pth --output-dir outputs/eval`. Export a manifest-aware demo video with `python scripts/media/export_demo_video.py --help`; it requires FFmpeg on `PATH` and a checkpoint whose saved `train_config` matches the requested manifest.

## Coding Style & Naming Conventions

Target Python 3.10+, use four-space indentation, type hints, and `pathlib.Path`. Follow `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for constants. Group imports as standard library, third-party, then local modules. Keep comments focused on CSI tensor shapes, normalization, and physical assumptions. Avoid unrelated refactors.

## Testing Guidelines

This handoff branch intentionally omits the historical test suite. Before modifying a retained workflow, add focused tests for the affected shape contract, manifest index rule, path validation, or checkpoint reconstruction. Use tiny synthetic fixtures and temporary directories; never require the full dataset in unit tests.

## Commit & Pull Request Guidelines

Recent history uses short imperative commits such as `Export report pose comparisons`. Keep commits scoped. Pull requests should summarize behavior, list commands run, state dataset and split assumptions, and include representative PNGs when visualization output changes. Do not include generated data or checkpoints.

## Security & Agent-Specific Instructions

Pass dataset locations through CLI arguments; never hard-code private server paths. Preserve unrelated user changes. Write repository-facing code and documentation in English and communicate with the user in Chinese unless requested otherwise. Use the `WiFiPose` environment, verify changes before claiming completion, and after project modifications commit and push the active `codex/` branch unless explicitly told not to.
