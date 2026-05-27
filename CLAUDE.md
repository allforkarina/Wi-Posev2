# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

**Local (Windows):**
```powershell
conda activate WiFiPose
```

**Linux server:**
```bash
# 1. Connect to server
ssh user@<server-ip>

# 2. Activate environment and pull latest code
conda activate WiFiPose
cd /path/to/Wi-Posev2
git pull origin main

# 3. Training (runs on GPU)
python train.py --dataset-root /data/WiFiPose/dataset/mmfi_pose_v3 \
    --source-envs lab --target-envs corridor \
    --epochs 50 --batch-size 64 --output-dir outputs/train_da

# 4. Evaluation with visualizations
python eval.py --dataset-root /data/WiFiPose/dataset/mmfi_pose_v3 \
    --checkpoint outputs/train_da/best_val_mpjpe.pth \
    --output-dir outputs/eval \
    --feature-viz --cross-env-viz \
    --source-env lab --target-env corridor \
    --output-format both

# 5. Download visualization outputs for local viewing
scp -r user@<server-ip>:/path/to/Wi-Posev2/outputs/eval/feature_viz/ .
```

Remote: `git@github.com:allforkarina/Wi-Posev2.git`, branch `main`.

## Architecture

Wi-Fi CSI-based single-frame human pose estimation. Input: CSI amplitude from 3 antennas `[B, 3, 114, 64]` (antenna × subcarrier × time). Output: OpenPose18 keypoints `[B, 18, 2]` normalized to `[-0.8, 0.8]`.

**Data flow:** `SpatialEncoder → AxialEncoder → Decoder → [B, 18, 2]`

- **`models/wiflow_spatial_encoder.py`**: CNN with symmetric residual downsampling blocks. `[B, 3, 114, 64] → [B, 128, 29, 16]`. Antenna mixer (1×1 conv) → feature stem → 3 resblocks (stride 2, 2, 1).
- **`models/wiflow_axial_encoder.py`**: 8-head multi-head attention along spatial (subcarrier) and temporal axes separately. 4 modes: `spatial_then_temporal`, `temporal_then_spatial`, `parallel_sum`, `parallel_concat`. 1×1 conv projects 128 → 256 channels. `[B, 128, 29, 16] → [B, 256, 29, 16]`.
- **Decoder** (one of three, selected by `--decoder-type`):
  - `joint` (default): 18 learnable joint queries cross-attend to spatial tokens, then GNN skeleton message passing + self-attention refinement → coordinate MLP head.
  - `hierarchical`: Staged coarse-to-fine retrieval (torso → limbs → face), each stage conditions on previous joint context.
  - `heatmap_msfn`: 3-stage PCM/PAF heatmap prediction with PAPM (pose-aware feature modulation) feedback. Argmax decoding on final stage PCM yields coordinates. Loss is PCM MSE + PAF MSE per stage.

- **`models/cece.py`**: Cross-Environment Channel Enhancement. Stateless module that computes per-channel cosine similarity between source/target domain-averaged features, reweights both domains' feature channels. Used only during DA training, between axial encoder and decoder.
- **`models/skeleton.py`**: OpenPose18 topology: 18 keypoints, 19 bone edges (including cross-body diagonals `(2,8)` and `(5,11)`), normalized adjacency for GNN.

**Domain adaptation training** (`train.py`): Dual-domain loop with CECE + ICAL. Splits `WiFlowModel.forward()` into spatial → CECE → decoder. Source domain gets all filtered data; target domain split by subject (80/20 train/val). Loss: `(source_supervised + target_supervised) / 2 + alpha * ICAL`. ICAL (Instance-level Consistency Alignment Loss) reweights feature-space distances by pose similarity. Alpha linearly warms up over `--ical-warmup-epochs`.

**Data layer**: `MemmapDataset` reads pre-built `.npy` memmap files (3 normalization variants), `ground_truth.npy`, and `meta.npz`. Collate function permutes NHWC → NCHW. Dataset is built via `scripts/build_memmap.py` from raw MM-Fi MAT files or via `scripts/build_groundtruth.py` from pre-annotated GT npy files.

**Evaluation** (`eval.py`): Restores model architecture from checkpoint's `train_config`, runs single-pass inference, outputs per-joint/per-action/per-environment CSVs. `--feature-viz` generates 6 research figures via forward hooks (`evaluation/hooks.py`).

## Commands

```powershell
# Run all tests
pytest

# Standard training
python train.py --dataset-root data/mmfi_pose --epochs 50 --batch-size 64 --output-dir outputs/train

# DA training (lab → corridor)
python train.py --dataset-root data/mmfi_pose --source-envs lab --target-envs corridor --epochs 50 --batch-size 64 --output-dir outputs/train_da

# CECE disabled (ICAL-only ablation)
python train.py --dataset-root data/mmfi_pose --source-envs lab --target-envs corridor --no-cece --output-dir outputs/train_da_no_cece

# Axial mode ablation
python train.py --dataset-root data/mmfi_pose --axial-mode parallel_sum --epochs 50 --batch-size 64 --output-dir outputs/train_parallel

# Decoder ablation
python train.py --dataset-root data/mmfi_pose --decoder-type hierarchical --epochs 50 --batch-size 64 --output-dir outputs/train_hier

# Quick sanity check
python train.py --dataset-root data/mmfi_pose --epochs 2 --batch-size 4 --output-dir outputs/sanity

# Evaluate a checkpoint
python eval.py --dataset-root data/mmfi_pose --checkpoint outputs/train/best_val_mpjpe.pth --output-dir outputs/eval

# Evaluate with feature visualizations
python eval.py --dataset-root data/mmfi_pose --checkpoint outputs/train/best_val_mpjpe.pth --output-dir outputs/eval --feature-viz --num-action-samples 3 --output-format both

# Build memmap dataset from raw MM-Fi files
python scripts/build_memmap.py --src /data/WiFiPose/dataset/dataset --dst /data/WiFiPose/dataset/mmfi_pose_v3 --train-subjects S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 --workers 8

# Build ground truth from pre-annotated npy files
python scripts/build_groundtruth.py --src /data/WiFiPose/dataset/ground_truth_npy --dst /data/WiFiPose/dataset/mmfi_pose_v3

# Visualize GT keypoints for a single frame
python scripts/visualize_gt.py --gt data/gt_merged/ground_truth.npy --frame 0

# Loss diagnosis
python scripts/diagnose_loss.py --dataset-root data/mmfi_pose
```

## Key conventions

- Python 3.10+, `from __future__ import annotations` in every module.
- Type hints on all function signatures.
- All model modules use `torch.nn`; no raw tensor operations in model code.
- Checkpoints store `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `best_metric`, and `train_config` (serialized `TrainConfig` dataclass).
- `tests/` is gitignored (local verification only). `conftest.py` adds project root to `sys.path`.
- Dataset artifacts (`*.npy`, `*.npz`, `*.h5`, `*.pth`, `*.pt`), virtual environments, and `outputs/` are gitignored.
- After each modification, commit and push to the remote:
  ```bash
  git add <changed files>
  git commit -m "<concise imperative message>"
  git push origin main
  ```
