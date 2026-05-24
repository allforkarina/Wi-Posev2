# Repository Guidelines

## Project Overview
Wi-Pose (v2): WiFi CSI-based single-frame 3D human pose estimation via OpenPose18 skeleton regression. The model takes CSI amplitude from three antennas and predicts 18 keypoint coordinates using attention-based architecture with structured skeleton priors.

Remote: `git@github.com:allforkarina/Wi-Posev2.git`

## Project Structure & Module Organization
- `dataloader.py`: Core module for loading NPY memmap datasets, providing `memmap_collate_fn` (NHWC→NCHW permutation), and factory functions `create_memmap_data_loader` / `create_memmap_data_loaders` for train/val/test splits.
- `data/memmap_dataset.py`: NPY memmap dataset reader with zero-copy OS-cached I/O. Loads CSI amplitude (3 normalization modes: `global_minmax`, `global_zscore`, `zscore`), OpenPose18 keypoints from `ground_truth.npy`, and metadata (action/subject/environment/frame_idx) from `meta.npz`. Supports environment/subject-based split construction and random val splitting.
- `data/heatmap_gt.py`: Functions for generating OpenPose18 PCM (Gaussian heatmaps) and PAF (bone vector fields) targets from normalized keypoint coordinates.
- `pose_targets.py`: Torch utilities for online OpenPose18 PCM/PAF target synthesis from normalized coordinates (`build_pcm_targets`, `build_paf_targets`, `build_pcm_paf_targets`), coordinate-to-heatmap mapping (`keypoints_to_heatmap_coords`), and argmax PCM decoding back to normalized keypoints (`decode_pcm_argmax`).
- `models/`: PyTorch model code — all modules use `from __future__ import annotations`.
  - `models/skeleton.py`: OpenPose18 skeleton topology — 18 keypoints, 19 bone edges (including cross-body diagonals `(2,8)` and `(5,11)`), and `build_normalized_adjacency()` for GNN-based skeleton refinement.
  - `models/wiflow_spatial_encoder.py`: CSI spatial encoder with antenna mixing (1×1 conv), feature stem, and symmetric time-frequency residual downsample blocks. Transforms `[B, 3, 114, 64]` → `[B, 128, 29, 16]`. Uses `SymmetricResidualDownsampleBlock` with 3×3 conv + 1×1 shortcut, stride-2 on both time and subcarrier axes.
  - `models/wiflow_axial_encoder.py`: Axial attention encoder with 4 modes: `spatial_then_temporal`, `temporal_then_spatial`, `parallel_sum`, `parallel_concat`. Applies 8-head multi-head attention along spatial (subcarrier) and temporal dimensions separately, then projects from 128 → 256 channels via 1×1 conv. Input `[B, 128, 29, 10]` → output `[B, 256, 29, 10]`.
  - `models/wiflow_joint_decoder.py`: Multi-layer joint cross-attention decoder. Maintains 18 learnable joint query embeddings that attend to spatial encoder tokens via 8-head cross-attention, followed by FFN and GNN-based skeleton message passing. Outputs `[B, 18, 2]` normalized coordinates.
  - `models/wiflow_hierarchical_joint_decoder.py`: Hierarchical joint decoder with staged retrieval — each stage retrieves one joint subset, attending to both spatial tokens and previously retrieved joint context. Outputs `[B, 18, 2]` normalized coordinates.
  - `models/wiflow_heatmap_decoder.py`: MultiFormer-style MSFN heatmap decoder with PAPM feedback. `WiFlowHeatmapDecoder` (one stage): shared 5-conv backbone → bottleneck → PCM + PAF heads (18 PCM + 38 PAF = 56 heatmap channels). `WiFlowPAPM`: pose-aware feature modulation using channel gate + spatial gate from previous-stage heatmaps. `WiFlowMSFNDecoder`: 3-stage progressive refinement with PAPM feedback, outputs `{"keypoints": tensor, "stages": list}`.
  - `models/wiflow_model.py`: End-to-end `WiFlowModel` orchestrating `spatial_encoder → axial_encoder → decoder` (one of `joint`, `hierarchical`, or `heatmap_msfn`). For heatmap decoder, applies `decode_pcm_argmax` on last-stage PCM for coordinate output.
- `train.py`: Training entrypoint. Provides `TrainConfig` dataclass, loss functions (`coord_l1 + bone_loss_weight * bone_l1` for direct decoders, multi-stage PCM/PAF MSE for heatmap decoders), metrics (MPJPE, PCK at thresholds 0.1–0.5), `OneCycleLR` scheduler with cosine annealing, gradient clipping, checkpointing (best_val_mpjpe, best_val_pck_0_2, last), and CSV logging to `train_log.csv`.
- `eval.py`: Evaluation entrypoint. Loads checkpoints (reconstructs model from saved `train_config`), runs single-pass inference on test split, outputs per-joint / per-action / per-environment CSVs. Supports `--feature-viz` flag for research-grade intermediate feature visualization.
- `evaluation/`: Evaluation pipeline package.
  - `evaluation/hooks.py`: Context manager (`WiFlowHookContext`, `wiflow_hooks`) for non-invasive forward hook registration on WiFlow submodules — collects intermediate feature outputs with automatic cleanup.
  - `evaluation/feature_viz.py`: Orchestrator and 6 figure-drawing functions: (1) antenna channel response, (2) resblock PCA trajectory, (3) axial attention maps, (4) joint query t-SNE + cosine similarity, (5) PCM/PAF heatmap quality radar chart, (6) feature-pose Pearson correlation landscape. Supports `png`/`pdf` output with customizable figure dimensions.
- `scripts/`: Preprocessing and diagnostic utilities.
  - `scripts/build_memmap.py`: Builds NPY memmap dataset from raw MM-Fi directory structure.
  - `scripts/build_groundtruth.py`: Builds ground-truth keypoint statistics and visualizations.
  - `scripts/visualize_gt.py`: Visualizes ground-truth pose annotations overlaid on video frames.
  - `scripts/diagnose_loss.py`: Standalone loss diagnostic for analyzing PCM prediction quality per joint.
- `tests/`: `pytest` unit tests (13 test files). Mirror module names: `test_dataloader.py`, `test_memmap_dataset.py`, `test_pose_targets.py`, `test_skeleton.py`, `test_wiflow_model.py`, `test_wiflow_spatial_encoder.py`, `test_wiflow_axial_encoder.py`, `test_wiflow_joint_decoder.py`, `test_wiflow_hierarchical_joint_decoder.py`, `test_wiflow_msfn_decoder.py`, `test_train.py`, `test_eval.py`. `conftest.py` adds project root to `sys.path`. Tests directory is in `.gitignore` (local use only).
- `docs/`: Planning documents and architecture analysis (memmap migration, model architecture, project architecture, superpowers plans).
- `.gitignore`: Excludes Python caches, environments, datasets (`*.npy`, `*.npz`, `*.h5`), checkpoints (`*.pth`, `*.pt`), outputs, and `tests/` directory.

## Data Flow
```
Raw CSI [B, 3, 114, 64] (channels-first: antenna × subcarrier × time)
  → spatial_encoder → [B, 128, 29, 16]
  → axial_encoder → [B, 256, 29, 10]
  → decoder → [B, 18, 2] normalized keypoints (or {"keypoints": ..., "stages": [...]})
```

The subcarrier axis (114) carries spatial-frequency response, the antenna axis (3) carries spatial phase-difference/direction information, and the temporal axis (64, upsampled from 10 original time shots) carries motion/Doppler cues.

## Project Domain Knowledge
- One CSI sample: `64 time steps × 3 antennas × 114 subcarriers`. Model input: `[B, 3, 114, 64]` (channels-first).
- Only CSI amplitude is used (3 channels, one per antenna). Phase information is not used.
- Target pose: OpenPose18 keypoint set (18 joints: Nose, Neck, RSh, RElb, RWr, LSh, LElb, LWr, RHip, RKnee, RAnk, LHip, LKnee, LAnk, REye, LEye, REar, LEar). 19 bone edges including cross-body diagonals `(2,8)` and `(5,11)`.
- CSI is a low-resolution, high-noise, implicit sensing signal. Strong skeleton priors are critical for accurate regression.
- Preserve CSI physical dimension semantics: avoid arbitrary flattening or pooling that mixes antenna/subcarrier/temporal meanings before attention-based selection.
- Prefer attention-based information selection over destructive pooling; use structured supervision (bone loss, PCM/PAF multi-stage) alongside coordinate loss.

## Build, Test, and Development Commands
Activate the Conda environment first:

```powershell
conda activate WiFiPose
pip install numpy scipy h5py tqdm torch pytest
```

Build an NPY memmap dataset:

```powershell
python scripts\build_memmap.py --dataset-root D:\path\to\raw\dataset --output-dir data\mmfi_pose --seed 42
```

Run tests (from project root):

```powershell
pytest
```

Domain adaptation (cross-environment) training:

```powershell
python train.py --dataset-root data\mmfi_pose --source-envs lab --target-envs corridor --epochs 50 --batch-size 64 --output-dir outputs\train_da
```

Disable CECE (ICAL-only ablation):

```powershell
python train.py --dataset-root data\mmfi_pose --source-envs lab --target-envs corridor --no-cece --output-dir outputs\train_da_no_cece
```

Adjust ICAL strength and warmup:

```powershell
python train.py --dataset-root data\mmfi_pose --source-envs lab --target-envs corridor --alpha 0.5 --ical-warmup-epochs 10 --output-dir outputs\train_da_alpha05
```

Decoder ablation with DA:

```powershell
python train.py --dataset-root data\mmfi_pose --source-envs lab --target-envs corridor --decoder-type hierarchical --output-dir outputs\train_da_hierarchical
```

Quick DA sanity check (2 epochs):

```powershell
python train.py --dataset-root data\mmfi_pose --source-envs lab --target-envs corridor --epochs 2 --batch-size 4 --output-dir outputs\sanity_da
```

Supported `--source-envs` / `--target-envs` values: environment names from the dataset (e.g., "lab", "corridor").

Default training:

```powershell
python train.py --dataset-root data\mmfi_pose --epochs 50 --batch-size 64 --output-dir outputs\train
```

Default config: CSI amplitude (3ch), `OneCycleLR` (cosine annealing, pct_start=0.3), gradient clipping (norm=1.0), `(source_coord_l1 + target_coord_l1) / 2 + (source_bone_l1 + target_bone_l1) / 2 * 0.5 + alpha * ICAL` for direct decoders, AdamW (lr=2e-5, max_lr=5e-4, weight_decay=5e-4). Default source="lab", target="corridor", alpha=0.1, ical_warmup_epochs=5.

Axial encoder ablation:

```powershell
python train.py --dataset-root data\mmfi_pose --axial-mode temporal_then_spatial --epochs 50 --batch-size 64 --output-dir outputs\train_temporal_then_spatial
```

Supported `--axial-mode` values: `spatial_then_temporal`, `temporal_then_spatial`, `parallel_sum`, `parallel_concat`.

Decoder ablation:

```powershell
python train.py --dataset-root data\mmfi_pose --decoder-type hierarchical --epochs 50 --batch-size 64 --output-dir outputs\train_hierarchical_decoder
python train.py --dataset-root data\mmfi_pose --decoder-type heatmap_msfn --epochs 50 --batch-size 64 --output-dir outputs\train_heatmap_msfn
```

Supported `--decoder-type` values: `joint`, `hierarchical`, `heatmap_msfn`.

The `heatmap_msfn` decoder uses 3 stages, 128 heatmap feature channels, 512 decoder hidden channels, PAPM feedback from concatenated PCM/PAF, and argmax decoding on last-stage PCM. Exposes `--heatmap-size` (default 36), `--heatmap-sigma` (default 1.5), `--paf-width` (default 1.0), `--paf-loss-weight` (default 1.0).

Evaluate one checkpoint:

```powershell
python eval.py --dataset-root data\mmfi_pose --checkpoint outputs\train\best_val_mpjpe.pth --output-dir outputs\eval
```

Evaluate with feature visualization:

```powershell
python eval.py --dataset-root data\mmfi_pose --checkpoint outputs\train\best_val_mpjpe.pth --output-dir outputs\eval --feature-viz --num-action-samples 3 --output-format both
```

## Coding Style & Naming Conventions
- Python 3.10+ with `from __future__ import annotations` at the top of every module.
- Type hints on function signatures; `pathlib.Path` for paths.
- Imports: standard library → third-party (`torch`, `numpy`) → local (project modules).
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, uppercase for constants (`NUM_OPENPOSE_KEYPOINTS`, `OPENPOSE_BONE_EDGES`, `PCK_THRESHOLDS`).
- 4-space indentation. Keep comments focused on shape contracts, normalization assumptions, and physical dimension semantics.
- Use Chinese for conversational replies; English for code, comments, and documentation.

## Testing Guidelines
- Framework: `pytest`. Files named `test_*.py`, functions named `test_<behavior>()`.
- `conftest.py` adds project root to `sys.path` so all test files can import project modules directly.
- Tests directory is gitignored (local verification only).
- Use temporary directories and tiny synthetic fixtures for datasets/models.
- Test coverage areas: split generation, path validation, shape validation, normalization, model shape contracts, PCM/PAF target synthesis, heatmap decoder stage outputs, memmap dataset loading.

## Training & Evaluation Outputs
- Output directory: `outputs/` by default (gitignored).
- Checkpoints: `best_val_mpjpe.pth`, `best_val_pck_0_2.pth`, `last.pth` — each stores `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `best_metric`, and `train_config` (serialized `TrainConfig` as dict).
- Training log: `train_log.csv` — appended per epoch with loss components, MPJPE, PCK at multiple thresholds, LR, epoch time.
- Evaluation: `per_joint_metrics.csv`, `per_action_metrics.csv`, `per_environment_metrics.csv`.
- Feature visualization: `.png`/`.pdf` figures grouped by action/environment samples.

## Commit & Push Guidelines
- Use concise imperative commit messages (e.g., `Add NPY memmap dataset support`).
- Remote: `git@github.com:allforkarina/Wi-Posev2.git` (branch: `main`).
- After each project modification, commit the change and push to the remote in the same turn unless the user explicitly asks not to push.
- Do not commit generated datasets, virtual environments, checkpoints, or `tests/` directory.

## Security & Configuration
- Do not hard-code dataset paths. Use `--dataset-root` CLI argument.
- Keep large datasets and sensitive files outside version control.

## Agent-Specific Instructions
- Before changing code, prefer the smallest working change; avoid unrelated refactors.
- After each project modification, commit and push to the configured GitHub remote unless explicitly told not to.
- Before running project code or tests, activate Conda environment: `conda activate WiFiPose`.
- Update this `AGENTS.md` file whenever project structure, commands, or workflows change.