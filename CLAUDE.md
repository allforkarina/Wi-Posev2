# Repository Guidelines

## Project Structure & Module Organization
- `dataloader.py`: Core module for loading NPY memmap datasets, creating PyTorch `DataLoader` instances with `memmap_collate_fn`, and providing `create_memmap_data_loader` / `create_memmap_data_loaders` / `create_few_shot_data_loader` factory functions.
- `data/memmap_dataset.py`: NPY memmap dataset reader that loads CSI amplitude, OpenPose18 keypoints, and metadata from `.npy`/`.npz` files with zero-copy OS-cached I/O.
- `data/heatmap_gt.py`: OpenPose18 coordinate conversion utilities (coco17_to_openpose18, valid_point).
- `pose_targets.py`: Reserved for future pose target utilities.
- `models/`: PyTorch model code, including the full WiFlow model, CSI spatial encoder with symmetric spatio-temporal downsampling, axial attention encoder, multi-layer joint cross-attention decoder, hierarchical joint decoder ablation, and shared OpenPose18 skeleton topology. The active single-frame model path is CSI amplitude input -> spatial encoder with antenna mixing, feature stem, and symmetric time-frequency residual blocks -> axial encoder -> the configured decoder.
- `train.py`: Root-level training entrypoint for WiFlow pose regression, including losses, metrics, optimizer, scheduler, checkpointing, and CSV logging. Supports `--mode source_only` (single-domain training) and `--mode finetune` (cross-domain few-shot finetuning with Tier 1 freeze).
- `eval.py`: Root-level evaluation entrypoint for loading checkpoints, computing test metrics, and optionally generating research-grade feature visualizations via `--feature-viz`. Supports `--eval-envs` (environment filtering) and `--exclude-indices` (exclude few-shot training frames).
- `evaluation/`: Evaluation pipeline package.
  - `evaluation/hooks.py`: Forward hook context manager (`WiFlowHookContext`, `wiflow_hooks`) for non-invasive intermediate feature extraction from WiFlow submodules.
  - `evaluation/feature_viz.py`: Orchestrator and figure-drawing functions for research-grade feature visualization (antenna channel response, resblock PCA trajectory, axial attention maps, joint query t-SNE, feature-pose correlation).
- `scripts/`: Preprocessing and diagnostic utilities.
  - `scripts/build_memmap.py`: Command-line wrapper that builds an NPY memmap dataset from the raw MM-Fi directory structure.
  - `scripts/build_groundtruth.py`: Builds ground-truth keypoint statistics and visualizations.
  - `scripts/visualize_gt.py`: Visualizes ground-truth pose annotations overlaid on corresponding video frames.
- `tests/`: `pytest` unit tests. Mirror module names such as `tests/test_dataloader.py`, `tests/test_wiflow_model.py`, or `tests/test_wiflow_decoder.py`.
- `.gitignore`: Excludes Python caches, local environments, generated datasets, checkpoints, and editor files from Git.

Generated datasets can be large and should not be committed. Keep raw dataset roots outside the repository.

## Project Domain Knowledge
- One CSI sample is a physical signal tensor shaped `64 time steps x 3 antennas x 114 subcarriers` in the NPY memmap dataset. The model input is `[B, 3, 114, 64]` (channels-first). The subcarrier axis carries spatial-frequency response, the antenna axis carries spatial phase-difference and direction information, and the temporal axis (64 steps, upsampled from 10 original time shots) carries motion cues such as Doppler effects.
- Only CSI amplitude is used as input (3 channels, one per antenna). Phase information is not used.
- The target pose is the structured OpenPose18 keypoint set (18 joints including neck). The 18 joints are not independent coordinates; they are constrained by the human skeleton topology (19 bone edges).
- The central modeling gap is that CSI is a low-resolution, high-noise, implicit sensing signal, while pose regression needs precise coordinates. Strong skeleton priors are important for bridging that gap.
- Preserve CSI physical dimension semantics where practical. Avoid arbitrary flattening or pooling that mixes antenna, subcarrier, and temporal meanings before the model has selected useful information.
- Prefer attention-based information selection over destructive pooling for low-SNR CSI features, and use structured supervision such as bone or topology-aware losses in addition to coordinate losses.

## Build, Test, and Development Commands
Use the existing Conda environment for development commands:

```powershell
conda activate WiFiPose
pip install numpy scipy h5py tqdm torch pytest
```

Build an NPY memmap dataset:

```powershell
python scripts\build_memmap.py --dataset-root D:\path\to\raw\dataset --output-dir data\mmfi_pose --seed 42
```

Run tests:

```powershell
pytest
```

Run a quick training sanity check:

```powershell
python train.py --mode source_only --dataset-root data\mmfi_pose --epochs 5 --subset-size 32 --output-dir outputs\sanity
```

Run the default training configuration:

```powershell
python train.py --mode source_only --dataset-root data\mmfi_pose --epochs 50 --batch-size 64 --output-dir outputs\train
```

The default training configuration uses CSI amplitude input (3 channels), `OneCycleLR`, gradient clipping, `coord_l1 + 0.5 * bone_l1`, the baseline axial mode `spatial_then_temporal`, and AdamW weight decay.

Run an axial-attention encoder ablation:

```powershell
python train.py --mode source_only --dataset-root data\mmfi_pose --axial-mode temporal_then_spatial --epochs 50 --batch-size 64 --output-dir outputs\train_temporal_then_spatial
```

Run a hierarchical decoder ablation:

```powershell
python train.py --mode source_only --dataset-root data\mmfi_pose --decoder-type hierarchical --epochs 50 --batch-size 64 --output-dir outputs\train_hierarchical_decoder
```

Supported `--axial-mode` values are `spatial_then_temporal`, `temporal_then_spatial`, `parallel_sum`, and `parallel_concat`. Supported `--decoder-type` values are `joint` and `hierarchical`. Checkpoints store the selected mode, decoder type, and settings in `train_config`, and evaluation rebuilds the model from that saved configuration.

Evaluate one checkpoint:

```powershell
python eval.py --dataset-root data\mmfi_pose --checkpoint outputs\train\best_val_mpjpe.pth --output-dir outputs\eval
```

### Cross-Domain Few-Shot Finetune Pipeline

```powershell
# Phase 1: Source-only Training
python train.py --mode source_only --dataset-root data\mmfi_pose --source-envs env1 --output-dir outputs\source_baseline --epochs 50

# Phase 2: Baseline Evaluation
python eval.py --dataset-root data\mmfi_pose --checkpoint outputs\source_baseline\best_val_mpjpe.pth --eval-envs env2 --output-dir outputs\baseline_eval

# Phase 3: Few-shot Finetune
python train.py --mode finetune --dataset-root data\mmfi_pose --target-envs env2 --output-dir outputs\finetune --finetune-from outputs\source_baseline\best_val_mpjpe.pth --few-shot-subjects 4 --few-shot-frames 5 --epochs 30

# Phase 4: Post-FT Evaluation
python eval.py --dataset-root data\mmfi_pose --checkpoint outputs\finetune\best_train_loss.pth --eval-envs env2 --output-dir outputs\finetune_eval --exclude-indices outputs\finetune\few_shot_train_indices.npy
```

## Coding Style & Naming Conventions
Use Python 3.10+ syntax, type hints, and `pathlib.Path` for paths. Group imports as standard library, third-party, then local. Follow existing naming: `snake_case` functions/variables, `PascalCase` classes, and uppercase constants such as `NUM_OPENPOSE_KEYPOINTS`. Use 4-space indentation. Keep comments focused on dataset assumptions, shapes, and normalization.

## Testing Guidelines
Automated tests use `pytest`. Add tests for split generation, path validation, shape validation, normalization edge cases, model shape contracts, and memmap dataset loading. Name files `test_*.py` and tests `test_<behavior>()`. Use temporary directories and tiny synthetic fixtures.

Training and evaluation outputs are written under `outputs/` by default. Checkpoints include `best_val_mpjpe.pth`, `best_val_pck_0_2.pth`, and `last.pth`; epoch metrics are appended to `train_log.csv`. Evaluation visualizations are saved as `.png` files grouped by action/environment samples.

## Commit & Pull Request Guidelines
This checkout has no `.git` history, so no convention can be inferred. Use concise imperative commits, for example `Add NPY memmap dataset support`. Pull requests should include a summary, commands run, dataset assumptions, and relevant shape or frame-count output. Do not commit generated datasets, virtual environments, or machine-specific paths.

## Security & Configuration Tips
Do not hard-code private dataset locations beyond documented defaults. Pass dataset paths with `--dataset-root` and keep large or sensitive data outside version control.

## Code Modification Constraints (Karpathy Guidelines)

Behavioral guidelines to reduce common LLM coding mistakes, derived from Andrej Karpathy's observations on LLM coding pitfalls.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Agent-Specific Instructions
Write repository-facing agent notes, documentation, and code comments in English. Keep comments neatly aligned with surrounding style. Use Chinese for conversational replies unless the user requests another language.

After each project modification, commit the change and push it to the configured GitHub remote in the same turn unless the user explicitly asks not to push.

Whenever the project undergoes significant modifications (architecture changes, new CLI arguments, new training modes, or modified workflows), update this `CLAUDE.md` file in the same turn to keep training/evaluation commands and module descriptions accurate.

Before changing code, apply the `karpathy-guidelines` skill: state assumptions when needed, prefer the smallest working change, avoid unrelated refactors, and verify the result with a concrete check.

Before running project code or tests, activate the existing Conda environment with `conda activate WiFiPose` to ensure commands run in the established project environment.