# REASONIX.md — Wi-Pose v2

## Stack
- Python 3.10+ with `from __future__ import annotations` everywhere
- PyTorch (AdamW, OneCycleLR, DataLoader)
- NumPy (memmap-backed .npy dataset I/O — no HDF5)
- pytest for testing
- Conda env `WiFiPose` (no pip requirements.txt or pyproject.toml)

## Layout
| Dir | Purpose |
| --- | --- |
| `train.py` | Source-only training + cross-domain finetuning entry point |
| `eval.py` | Metrics (PCK, MPJPE), CSV exports, CSI/skeleton PNG visualisations |
| `models/` | WiFlow encoder/decoder, CSI calibration, feature bank, skeleton, wrist refiner |
| `data/` | Memmap dataset + split manifest (deterministic train/val/test splits) |
| `dataloader.py` | DataLoader factories + `memmap_collate_fn` (CSI permute channels-last → channels-first) |
| `evaluation/` | Feature viz, pose comparison PNGs, hook context |
| `experiments/` | Report-suite experiment task definitions |
| `scripts/` | Preprocessing, benchmarking, export, diagnostics — each is a standalone CLI |
| `tests/` | pytest tests mirroring module structure; `test_*.py` files, `test_<behavior>()` functions |

## Commands
```bash
conda activate WiFiPose
pytest
python train.py --mode source_only --dataset-root data/mmfi_pose --epochs 1 --subset-size 32 --output-dir outputs/sanity
python eval.py --dataset-root data/mmfi_pose --checkpoint outputs/train/best_val_pck_0_2.pth --output-dir outputs/eval
python scripts/export_report_pose_visualizations.py --help
```

## Conventions
- **Imports:** stdlib → third-party → local; `sys.path.insert(0, str(ROOT))` in scripts (no package install)
- **Naming:** `snake_case` fns/vars, `PascalCase` classes, `UPPER_CASE` constants
- **Types:** type hints on all function signatures; `pathlib.Path` over `str` for paths
- **CSI shapes:** memmap stores `(B, H, W, C)`; `memmap_collate_fn` permutes to `(B, C, H, W)` for the model
- **Commits:** short imperative style ("Export report pose comparisons"), scoped to one change
- **Testing:** no conftest.py; tests use `tempfile.mkdtemp` fixtures and synthetic tensors — never full dataset

## Watch out for
- No build system / package install — every script manually adds the project root to `sys.path`; relative imports won't work outside that pattern
- `outputs/`, `checkpoints/`, `datasets/`, `*.pth`, `*.npy` are all gitignored — generated artifacts belong there
- Train checkpoints store a `train_config` dict alongside `model_state_dict`; `eval.py` reads it to reconstruct the model architecture before loading weights
