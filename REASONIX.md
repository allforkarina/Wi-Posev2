# REASONIX.md — WiFlow Pose Estimation

## Stack
- Python 3.10+, PyTorch (torch, torch.nn, torch.optim)
- NumPy for array ops, NPY memmap dataset I/O
- OpenPose18 skeleton: 18 keypoints, 19 bone edges
- Conda env `WiFiPose` (torch, numpy, scipy, h5py, tqdm, pytest)
- No package manifest — scripts run directly from repo root

## Layout
- `train.py` — training entrypoint (source-only + cross-domain few-shot finetune)
- `eval.py` — evaluation entrypoint (metrics, CSV, pose/feature viz)
- `dataloader.py` — NPY memmap DataLoader factories and `memmap_collate_fn`
- `data/` — `memmap_dataset.py` (Dataset), `heatmap_gt.py` (coordinate utils)
- `models/` — `WiFlowModel`, spatial encoder, axial encoder, joint/hierarchical decoders, skeleton
- `evaluation/` — forward hooks, feature viz, pose viz
- `scripts/` — `build_memmap.py`, `build_groundtruth.py`, `visualize_gt.py`
- `outputs/` — training/eval artifacts (gitignored)
- `pose_targets.py` — reserved, currently empty
- `docs/` — planning docs and specs

## Commands
```powershell
# Build memmap dataset
python scripts\build_memmap.py --dataset-root D:\path\to\raw\dataset --output-dir data\mmfi_pose --seed 42

# Train
python train.py --mode source_only --dataset-root data\mmfi_pose --epochs 50 --batch-size 64 --output-dir outputs\train
# Evaluate
python eval.py --dataset-root data\mmfi_pose --checkpoint outputs\train\best_val_mpjpe.pth --output-dir outputs\eval

# Few-shot cross-domain
python train.py --mode source_only --dataset-root data\mmfi_pose --source-envs env1 --output-dir outputs\source --epochs 50
python train.py --mode finetune --dataset-root data\mmfi_pose --target-envs env2 --output-dir outputs\finetune --finetune-from outputs\source\best_val_mpjpe.pth --few-shot-subjects 4 --few-shot-frames 5 --epochs 30
python eval.py --dataset-root data\mmfi_pose --checkpoint outputs\finetune\best_train_loss.pth --eval-envs env2 --output-dir outputs\ft_eval --exclude-indices outputs\finetune\few_shot_train_indices.npy

# Tests
pytest
```

## Conventions
- `from __future__ import annotations` at top of every module
- `snake_case` functions/variables, `PascalCase` classes, `UPPER_CASE` constants
- Type hints on all signatures; `pathlib.Path` for paths
- Imports: stdlib → third-party → local; 4-space indent
- Public API via `__all__` in package `__init__.py`

## Constraints
- After every project update, commit and push changes to GitHub.
- All code modifications must follow the `karpathy-guidelines` skill.

## Watch out for
- No installable package (`pyproject.toml`/`setup.py`) — run scripts from repo root.
- `tests/` is gitignored and does not currently exist; the CLAUDE.md references test files that need creating.
- CSI shape: raw memmap `[N,64,3,114]` → collate permutes to `[B,3,114,64]` (antenna, subcarrier, time). Amplitude only; phase discarded.
- `.npy`/`.npz`/`.pt`/`.pth` artifacts are gitignored.
- `--axial-mode`: `spatial_then_temporal`, `temporal_then_spatial`, `parallel_sum`, `parallel_concat`. `--decoder-type`: `joint`, `hierarchical`.
