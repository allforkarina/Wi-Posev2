# Single-Domain Regression + Cross-Domain Few-shot Finetune Design

**Date**: 2026-05-27
**Status**: Approved
**Repo**: `git@github.com:allforkarina/Wi-Posev2.git` (rebased from Wi_Pose single-domain baseline)

## Decisions

| # | Decision | Conclusion |
|---|----------|------------|
| 1 | Split strategy | Subject-level (keep existing), no frame-level change |
| 2 | Training modes | `source_only` + `finetune` only, no DA |
| 3 | AMP | Not added, keep training loop simple |
| 4 | `--mode` | Required explicitly (no default) |
| 5 | Few-shot defaults | 4 subjects × 5 frames = 540 frames |
| 6 | Freeze strategy | Tier 1 only (BN/LN affine + joint_queries + coordinate_head) |
| 7 | Heatmap | Delete heatmap-specific code, keep generic utilities (COCO17→OP18) |
| 8 | `"all"` split | Internal-only in dataloader/MemmapDataset, not exposed to user |
| 9 | `--eval-split` | Removed, eval always uses full target-domain data |

## Files to Delete

| File | Reason |
|------|--------|
| `models/wiflow_heatmap_decoder.py` | Pure heatmap MSFN decoder, no longer used |
| `scripts/diagnose_loss.py` | PCM prediction diagnostics, entirely heatmap-specific |
| `docs/models_architecture_analysis.md` | Old v2 architecture doc with DA and heatmap references |

## Files to Modify

### 1. `data/heatmap_gt.py`

- Remove: `build_pcm_targets`, `build_paf_targets`, `OPENPOSE_18_NAMES`, `heatmap_to_pose_coords`
- Keep: `coco17_to_openpose18`, `valid_point` (used by `build_groundtruth.py`)

### 2. `pose_targets.py`

- Remove: `build_pcm_paf_targets`, `build_pcm_targets`, `decode_pcm_argmax`
- Keep: coordinate normalization utilities if any

### 3. `data/memmap_dataset.py`

- Add `_sample_few_shot(subjects, frames_per_action)`: deterministic few-shot sampling
  - Sort subjects by ID, take first K
  - For each (action, subject) group: sort frames, use `np.linspace` to uniformly sample N frames
  - Return deterministic index list
- Add `"all"` to internal split names (skip random split, return sorted candidate_indices)
- Existing `_build_split` unchanged (already subject-level 8:2)

### 4. `dataloader.py`

- `create_memmap_data_loader`: add `envs: tuple[str, ...] | None = None` parameter
- Add `create_few_shot_data_loader(root, target_envs, few_shot_subjects, few_shot_frames, batch_size, num_workers)`:
  - Load target env full data (`split="all"`)
  - `_sample_few_shot` → `train_indices`
  - Complement (full − train_indices) → val_loader
  - Return `(train_loader, val_loader, train_indices)`
- Add `"all"` to `ALL_SPLITS` dictionary

### 5. `models/wiflow_model.py`

- Remove: `from .wiflow_heatmap_decoder import WiFlowMSFNDecoder`
- Remove: `from pose_targets import decode_pcm_argmax`
- Remove: `DECODER_TYPES = ("joint", "hierarchical", "heatmap_msfn")` → `("joint", "hierarchical")`
- Remove: `heatmap_size` parameter from `__init__`
- Remove: `self.heatmap_size = heatmap_size`
- Remove: `elif decoder_type == "heatmap_msfn"` branch in decoder selection
- Remove: decoder-type dispatch in `forward`: delete the `heatmap_msfn` path that calls `decode_pcm_argmax`

### 6. `train.py` — Major Rework

**New CLI arguments:**

```
--mode {source_only, finetune}   (required)
--source-envs env1 [env1 ...]    (for source_only)
--target-envs env1 [env1 ...]    (for finetune)
--finetune-from PATH             (required for finetune)
--few-shot-subjects 4
--few-shot-frames 5
--freeze-tier 1                  (only Tier 1 supported)
```

**Removed CLI arguments (heatmap-related):**

```
--heatmap-size, --heatmap-sigma, --paf-width, --paf-loss-weight
--decoder-type heatmap_msfn
```

**Code structure changes:**

- `TrainConfig`: remove `heatmap_size`, `heatmap_sigma`, `paf_width`, `paf_loss_weight`; add `mode`, `finetune_from`, `few_shot_subjects`, `few_shot_frames`, `freeze_tier`; add `source_envs`, `target_envs`
- `compute_losses`: remove `heatmap_size/sigma/paf_width/paf_loss_weight` parameters and entire heatmap/msfn branch; keep only coordinate regression loss path
- `run_epoch` → rename to `run_source_only_epoch`, remove heatmap-specific metric returns (pcm_loss, paf_loss)
- Add `run_finetune_epoch(model, train_loader, optimizer, scheduler, config)`:
  - No per-epoch val
  - Save `epoch_XXX.pth` every epoch
  - Track best by `train_loss`
- Add `apply_finetune_tier(model, tier=1)`:
  - Freeze all params except those matching: `norm`, `bn`, `ln`, `joint_queries`, `coordinate_head`
  - Print trainable parameter count
- `run_training`: dispatch based on `config.mode`
- `main`: parse `--mode`, route to appropriate loader setup

**Hyperparams:** keep existing baseline:
- `lr=2e-5`, `max_lr=5e-4`, `weight_decay=5e-4`
- `dropout=0` (keep model defaults), `pct_start=0.3`
- `batch_size=64`, `epochs=50` (source_only) / `epochs=30` (finetune default)
- No early stopping in finetune mode

### 7. `eval.py`

**Removed:**
- `--eval-split` parameter (always use `split="all"` internally)

**Added:**
- `--eval-envs env1 [env1 ...]`: filter by environment
- `--exclude-indices PATH.npy`: load index array, exclude from evaluation

**Changed:**
- Remove `heatmap_size` from checkpoint loading logic
- `DECODER_TYPES` → only `("joint", "hierarchical")`
- Hardcode `split="all"` when constructing MemmapDataset

### 8. `evaluation/feature_viz.py`

- Delete `_fig5a_pcm_radar` function
- Delete `_fig5b_paf_direction_consistency` function
- Remove references to these figures from the orchestrator
- Remove `decoder_type == "heatmap_msfn"` conditionals

### 9. `scripts/build_groundtruth.py`

- Update imports from `data.heatmap_gt` (remove PCM/PAF function imports after cleanup)

### 10. `AGENTS.md`

- Remove heatmap-related CLI arguments from command examples
- Remove `heatmap_msfn` decoder description
- Remove DA-related sections
- Update training command examples to reflect `--mode source_only` / `--mode finetune`
- Remove `--decoder-type heatmap_msfn` from ablation commands

## Unchanged Files

`models/wiflow_spatial_encoder.py`, `models/wiflow_axial_encoder.py`, `models/wiflow_joint_decoder.py`, `models/wiflow_hierarchical_joint_decoder.py`, `models/skeleton.py`, `evaluation/hooks.py`, `scripts/build_memmap.py`, `scripts/visualize_gt.py`, all test files.

## Training Pipeline

```bash
# Phase 1: Source-only Training
python train.py --mode source_only --dataset-root mmfi_pose_v4 \
    --source-envs env1 --output-dir runs/source_baseline --epochs 50

# Phase 2: Baseline Evaluation
python eval.py --dataset-root mmfi_pose_v4 \
    --checkpoint runs/source_baseline/best_val_mpjpe.pth \
    --eval-envs env2 --output-dir runs/baseline_eval

# Phase 3: Few-shot Finetune
python train.py --mode finetune --dataset-root mmfi_pose_v4 \
    --target-envs env2 --output-dir runs/finetune \
    --finetune-from runs/source_baseline/best_val_mpjpe.pth \
    --few-shot-subjects 4 --few-shot-frames 5 --epochs 30

# Phase 4: Post-FT Evaluation
python eval.py --dataset-root mmfi_pose_v4 \
    --checkpoint runs/finetune/epoch_005.pth \
    --eval-envs env2 --output-dir runs/finetune_eval \
    --exclude-indices runs/finetune/few_shot_train_indices.npy
```