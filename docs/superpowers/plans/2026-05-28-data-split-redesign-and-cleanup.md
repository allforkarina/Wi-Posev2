# Data Split Redesign & Code Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign data splitting to subject-level (8:2 train/val), implement deterministic few-shot sampling with uniform frame spacing, make test set = target env all minus few-shot train shared between baseline and finetune eval, and remove dead parameters/unused code across the project.

**Architecture:** `_build_split` shuffles subjects (not frames) and splits 8:2. `_sample_few_shot` deterministically picks first K subjects by sorted ID and uniformly samples frames via linspace. `create_few_shot_data_loader` returns train + val (complement), with train indices saved to `.npy` for eval exclusion. Finetune mode skips per-epoch val, saves all checkpoints. Cleanup removes `time_packets`/`subcarrier_mode`/`build_targets` from MemmapDataset, unused factories from dataloader.py, and wires broken CLI→code connections.

**Tech Stack:** Python 3.10+, numpy, torch, existing MemmapDataset/DataLoader infra.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `data/memmap_dataset.py` | Subject-level split, deterministic few-shot, remove dead params |
| `dataloader.py` | Rewrite `create_few_shot_data_loader`, remove unused factories, extend `create_memmap_data_loader` |
| `train.py` | Save few-shot indices, skip finetune val, wire `compute_losses` params |
| `eval.py` | Wire `--eval-split`, add `--exclude-indices` |
| `CLAUDE.md` | Update all commands |

---

### Task 1: Subject-Level Split + Remove Dead Params in MemmapDataset

**Files:**
- Modify: `data/memmap_dataset.py`

- [ ] **Step 1: Remove dead parameters (`time_packets`, `subcarrier_mode`, `build_targets`)**

Remove from `__init__` signature:
```python
# REMOVE these params:
# time_packets: int = 64,
# subcarrier_mode: str = "keep",
# build_targets: bool = True,
```

Remove `self.build_targets = build_targets` assignment.

Remove the `build_pcm_paf` import at line 11:
```python
# REMOVE: from data.heatmap_gt import build_pcm_paf
```

Remove the `build_targets` conditional block in `__getitem__` (lines 182-191):
```python
# REMOVE:
# if self.build_targets:
#     pcm, paf = build_pcm_paf(
#         kpts18, ...
#     )
#     item["pcm"] = torch.from_numpy(pcm)
#     item["paf"] = torch.from_numpy(paf)
```

Update all callers to remove `build_targets=False` argument:
- `dataloader.py:49,103,125,162,178`
- `train.py:550,556`

- [ ] **Step 2: Rewrite `_build_split` for subject-level 8:2 split**

Replace the current frame-level split with subject-level split:

```python
def _build_split(
    self,
    split: str,
    envs: Iterable[str] | None,
    train_subjects: Iterable[str] | None,
    test_subjects: Iterable[str] | None,
    random_val_ratio: float,
    seed: int,
) -> np.ndarray:
    num_total = len(self._samples)
    env_list = [str(e) for e in self._envs]
    sample_list = [str(s) for s in self._samples]

    env_set = set(envs) if envs else None
    subject_set = set(train_subjects) if train_subjects else None

    candidate_indices: list[int] = []
    for i in range(num_total):
        if env_set is not None and env_list[i] not in env_set:
            continue
        if subject_set is not None and sample_list[i] not in subject_set:
            continue
        candidate_indices.append(i)

    if split == "all":
        return np.asarray(sorted(candidate_indices), dtype=np.int64)

    # Subject-level split: group by subject, shuffle subjects, then 8:2
    rng = random.Random(seed)
    grouped: dict[str, list[int]] = {}
    for idx in candidate_indices:
        grouped.setdefault(sample_list[idx], []).append(idx)

    subject_ids = sorted(grouped.keys())
    rng.shuffle(subject_ids)
    n_val = max(1, int(len(subject_ids) * random_val_ratio))
    val_subjects = set(subject_ids[:n_val])
    train_subjects = set(subject_ids[n_val:])

    train_indices: list[int] = []
    val_indices: list[int] = []
    for subject, indices in sorted(grouped.items()):
        if subject in train_subjects:
            train_indices.extend(indices)
        else:
            val_indices.extend(indices)

    if split == "train":
        return np.asarray(sorted(train_indices), dtype=np.int64)
    else:
        return np.asarray(sorted(val_indices), dtype=np.int64)
```

- [ ] **Step 3: Rewrite `_sample_few_shot` for deterministic subject selection + uniform frame spacing**

```python
def _sample_few_shot(
    self,
    indices: np.ndarray,
    few_shot_frames: int,
    few_shot_subjects: int,
    seed: int,
) -> np.ndarray:
    """Deterministic few-shot sampling.

    Subjects: sorted by ID, take first ``few_shot_subjects``.
    Frames: per (action, subject) group, sort frames by index,
    then uniformly sample ``few_shot_frames`` frames via linspace.
    """
    if few_shot_frames <= 0 and few_shot_subjects <= 0:
        return indices

    # Group by (action, subject)
    grouped: dict[tuple[str, str], list[int]] = {}
    for idx in indices:
        idx_int = int(idx)
        key = (str(self._actions[idx_int]), str(self._samples[idx_int]))
        grouped.setdefault(key, []).append(idx_int)

    # Subject selection: first K by sorted ID (deterministic)
    if few_shot_subjects > 0:
        all_subjects = sorted(set(k[1] for k in grouped))
        chosen_subjects = set(all_subjects[:few_shot_subjects])
        grouped = {k: v for k, v in grouped.items() if k[1] in chosen_subjects}

    # Frame selection: uniform spacing via linspace over sorted frames
    result: list[int] = []
    for (_action, _subject), frame_indices in sorted(grouped.items()):
        if few_shot_frames > 0:
            sorted_frames = sorted(frame_indices)
            n = min(few_shot_frames, len(sorted_frames))
            linspace_indices = np.linspace(0, len(sorted_frames) - 1, n, dtype=int)
            sampled = [sorted_frames[i] for i in linspace_indices]
        else:
            sampled = frame_indices
        result.extend(sampled)

    return np.asarray(sorted(result), dtype=np.int64)
```

- [ ] **Step 4: Update `__init__` to remove `build_targets`, `time_packets`, `subcarrier_mode` from signature**

New signature (remove the 3 dead params):
```python
def __init__(
    self,
    data_dir: str | Path,
    split: str = "train",
    envs: Iterable[str] | None = None,
    train_subjects: Iterable[str] | None = None,
    test_subjects: Iterable[str] | None = None,
    random_val_ratio: float = 0.2,
    seed: int = 42,
    normalize: str = "global_minmax",
    heatmap_size: int = 36,
    heatmap_sigma: float = 1.5,
    paf_width: float = 1.0,
    pose_range: tuple[float, float] = (-0.8, 0.8),
    few_shot_frames: int = 0,
    few_shot_subjects: int = 0,
) -> None:
```

- [ ] **Step 5: Commit**

```bash
git add data/memmap_dataset.py
git commit -m "refactor: subject-level split, deterministic few-shot, remove dead params"
```

---

### Task 2: Clean Up dataloader.py + Rewrite few-shot Loader

**Files:**
- Modify: `dataloader.py`

- [ ] **Step 1: Remove unused factory and debug code**

Remove `create_memmap_data_loaders` (lines 64-79), `parse_args` (lines 204-208), `main` (lines 211-229), and `if __name__ == "__main__"` block (line 228-229).

Remove `import argparse` (line 5) — only used by removed debug code.

`import random` is still used by `create_da_data_loaders` and `create_few_shot_data_loader` (subject shuffle). Keep it.

- [ ] **Step 2: Add `"all"` to `SPLIT_NAMES`**

```python
SPLIT_NAMES = ("train", "val", "test", "all")
```

- [ ] **Step 3: Rewrite `create_few_shot_data_loader`**

Val = full complement (target env all minus few-shot train). Returns train_indices.

```python
def create_few_shot_data_loader(
    data_dir: str | Path,
    envs: Sequence[str],
    batch_size: int,
    few_shot_frames: int = 5,
    few_shot_subjects: int = 4,
    num_workers: int = 0,
    seed: int = 42,
) -> dict:
    """Create train/val DataLoaders for few-shot fine-tuning.

    Train: few-shot sampled subset of *envs* (≤ few_shot_subjects subjects,
    ≤ few_shot_frames per action×subject via deterministic uniform spacing).

    Val: ALL data in *envs* excluding the few-shot train indices.

    Returns dict with keys: ``"train"``, ``"val"``, ``"train_indices"``.
    """
    train_dataset = MemmapDataset(
        data_dir=data_dir, split="all", envs=list(envs),
        seed=seed,
        few_shot_frames=few_shot_frames,
        few_shot_subjects=few_shot_subjects,
    )
    train_indices = train_dataset.indices.copy()

    # Val = full target domain minus few-shot train
    full_dataset = MemmapDataset(
        data_dir=data_dir, split="all", envs=list(envs),
        seed=seed,
    )
    train_idx_set = set(int(i) for i in train_indices)
    val_indices = np.asarray(sorted(
        [i for i in full_dataset.indices if int(i) not in train_idx_set]
    ), dtype=np.int64)

    val_dataset = MemmapDataset(
        data_dir=data_dir, split="all", envs=list(envs),
        seed=seed,
    )
    val_dataset.indices = val_indices

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=memmap_collate_fn,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=memmap_collate_fn,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
    return {"train": train_loader, "val": val_loader, "train_indices": train_indices}
```

- [ ] **Step 4: Remove `build_targets=False` from all `MemmapDataset(...)` calls**

Remove every `build_targets=False` kwarg from `dataloader.py`.

- [ ] **Step 5: Remove unused `import random` if possible**

`create_da_data_loaders` does NOT use `random`. `create_few_shot_data_loader` (new version) also does NOT use `random`. Check: `create_memmap_data_loader` does not use it. So `import random` can be removed.

- [ ] **Step 6: Commit**

```bash
git add dataloader.py
git commit -m "refactor: rewrite few-shot loader, remove unused factories and debug code"
```

---

### Task 3: Fix eval.py — Wire --eval-split, Add --exclude-indices

**Files:**
- Modify: `eval.py`

- [ ] **Step 1: Add `import numpy as np` at top**

```python
import numpy as np
```

- [ ] **Step 2: Wire `--eval-split` to `create_memmap_data_loader`**

Replace line 296 `split="test"` with `split=args.eval_split`:

```python
test_loader = create_memmap_data_loader(
    data_dir=args.dataset_root,
    split=args.eval_split,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False,
    envs=args.eval_envs,
)
```

- [ ] **Step 3: Add `--exclude-indices` CLI arg and wiring**

Add after `--eval-split` arg (after line 284):

```python
parser.add_argument(
    "--exclude-indices", default=None,
    help="Path to .npy file with frame indices to exclude from eval.",
)
```

In `main()`, after loader creation (after line 301), add exclusion logic:

```python
if args.exclude_indices is not None:
    exclude_path = Path(args.exclude_indices)
    if not exclude_path.exists():
        raise FileNotFoundError(f"Exclude indices file not found: {exclude_path}")
    exclude_set = set(np.load(str(exclude_path)).astype(int).tolist())
    before = len(test_loader.dataset.indices)
    test_loader.dataset.indices = np.asarray(sorted(
        [i for i in test_loader.dataset.indices if int(i) not in exclude_set]
    ), dtype=np.int64)
    print(f"Excluded {before - len(test_loader.dataset.indices)} few-shot train samples, "
          f"{len(test_loader.dataset.indices)} remaining for evaluation.")
```

- [ ] **Step 4: Commit**

```bash
git add eval.py
git commit -m "fix: wire --eval-split, add --exclude-indices to eval.py"
```

---

### Task 4: Fix train.py — Save Few-Shot Indices, Skip Finetune Val, Wire compute_losses

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Wire `heatmap_sigma`, `paf_width`, `paf_loss_weight` to `compute_losses` calls**

In `run_da_epoch` (around line 290-299), add the three params:
```python
losses_s = compute_losses(
    y_s, kp_s_gt,
    bone_loss_weight=criterion_config.bone_loss_weight,
    heatmap_sigma=criterion_config.heatmap_sigma,
    paf_width=criterion_config.paf_width,
    paf_loss_weight=criterion_config.paf_loss_weight,
)
losses_t = compute_losses(
    y_t, kp_t_gt,
    bone_loss_weight=criterion_config.bone_loss_weight,
    heatmap_sigma=criterion_config.heatmap_sigma,
    paf_width=criterion_config.paf_width,
    paf_loss_weight=criterion_config.paf_loss_weight,
)
```

In `run_val_epoch` (around line 396-400):
```python
losses = compute_losses(
    prediction, target,
    bone_loss_weight=criterion_config.bone_loss_weight,
    heatmap_sigma=criterion_config.heatmap_sigma,
    paf_width=criterion_config.paf_width,
    paf_loss_weight=criterion_config.paf_loss_weight,
)
```

In `run_single_domain_epoch` (around line 435-438):
```python
losses = compute_losses(
    prediction, target,
    bone_loss_weight=criterion_config.bone_loss_weight,
    heatmap_sigma=criterion_config.heatmap_sigma,
    paf_width=criterion_config.paf_width,
    paf_loss_weight=criterion_config.paf_loss_weight,
)
```

- [ ] **Step 2: Save few-shot train indices in `_setup_finetune`**

Update `_setup_finetune` to return train_indices. Change return type:

```python
def _setup_finetune(
    config: TrainConfig, device: torch.device,
) -> tuple[WiFlowModel, DataLoader, DataLoader, np.ndarray]:
```

Add `import numpy as np` at top of train.py (check: not currently imported).

At the end of `_setup_finetune`:
```python
train_indices = loaders["train_indices"]
print(f"Few-shot target: {len(loaders['train'].dataset)} train / "
      f"{len(loaders['val'].dataset)} val samples")
return model, loaders["train"], loaders["val"], train_indices
```

Update `run_training` finetune branch to unpack the 4th return value and save:
```python
elif config.mode == "finetune":
    model, train_loader, val_loader, fs_indices = _setup_finetune(config, device)
    ...
    np.save(str(output_dir / "few_shot_train_indices.npy"), fs_indices)
    print(f"Saved few-shot train indices ({len(fs_indices)} frames) to "
          f"{output_dir / 'few_shot_train_indices.npy'}")
```

- [ ] **Step 3: Skip per-epoch val in finetune mode, save every checkpoint**

In `run_training` finetune branch loop, change to save checkpoint every epoch without val:

```python
# In finetune loop (after epoch training):
# Save checkpoint every epoch (no val during finetune)
save_checkpoint(
    output_dir / f"epoch_{epoch:03d}.pth",
    model, optimizer, scheduler, epoch,
    best_metric=train_metrics["loss"], config=config,
)
# Also update last.pth
save_checkpoint(
    output_dir / "last.pth",
    model, optimizer, scheduler, epoch,
    best_metric=train_metrics["loss"], config=config,
)
```

The print line for finetune should show train loss only (no val metrics):
```python
print(
    f"epoch={epoch:03d} "
    f"loss={train_metrics['loss']:.6f} "
    f"lr={current_lr:.2e} "
    f"epoch_time={epoch_time:.1f}s"
)
```

Remove `do_val` logic and early stopping from finetune mode (not applicable without val).

- [ ] **Step 4: Remove `build_targets=False` from `_setup_loaders_source_only` MemmapDataset calls**

```python
train_dataset = MemmapDataset(
    data_dir=config.dataset_root, split="train",
    envs=list(config.source_envs), seed=config.seed,
)
val_dataset = MemmapDataset(
    data_dir=config.dataset_root, split="val",
    envs=list(config.source_envs), seed=config.seed,
)
```

- [ ] **Step 5: Commit**

```bash
git add train.py
git commit -m "fix: wire compute_losses params, save few-shot indices, skip val in finetune"
```

---

### Task 5: Clean Up Dead Code in heatmap_gt.py

**Files:**
- Modify: `data/heatmap_gt.py`

- [ ] **Step 1: Remove unused names**

Remove `OPENPOSE_18_NAMES` (line 8) — defined but never referenced.

Remove `heatmap_to_pose_coords` function (lines 123-133) — defined but never called.

- [ ] **Step 2: Commit**

```bash
git add data/heatmap_gt.py
git commit -m "refactor: remove unused OPENPOSE_18_NAMES and heatmap_to_pose_coords"
```

---

### Task 6: Update CLAUDE.md Commands

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update all commands with new CLI flags**

```markdown
# Phase 1: Source-only training
python train.py --dataset-root data/mmfi_pose --mode source_only \
    --source-envs env1 --epochs 50 --batch-size 128 \
    --output-dir outputs/train_source

# Phase 2: Baseline eval (source checkpoint on ALL target domain data)
python eval.py --dataset-root data/mmfi_pose \
    --checkpoint outputs/train_source/best_val_mpjpe.pth \
    --eval-split all --eval-envs env2 \
    --output-dir outputs/eval_baseline

# Phase 3: Tier 1 fine-tune (few-shot target, no val during training)
python train.py --dataset-root data/mmfi_pose --mode finetune \
    --finetune-from outputs/train_source/best_val_mpjpe.pth \
    --target-envs env2 --few-shot-frames 5 --few-shot-subjects 4 \
    --freeze-tier 1 --epochs 30 --batch-size 128 --finetune-lr 1e-5 \
    --output-dir outputs/finetune_tier1

# Phase 3 eval: evaluate finetuned model excluding few-shot train samples
python eval.py --dataset-root data/mmfi_pose \
    --checkpoint outputs/finetune_tier1/epoch_030.pth \
    --eval-split all --eval-envs env2 \
    --exclude-indices outputs/finetune_tier1/few_shot_train_indices.npy \
    --output-dir outputs/eval_finetune_tier1
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update commands for new data split and eval flags"
```

---

### Task 7: Verification

**Files:** None (manual verification)

- [ ] **Step 1: Verify subject-level 8:2 split**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from data.memmap_dataset import MemmapDataset
ds_train = MemmapDataset('data/mmfi_pose', split='train', envs=['env1'], seed=42)
ds_val = MemmapDataset('data/mmfi_pose', split='val', envs=['env1'], seed=42)
train_subjects = set(str(ds_train._samples[int(i)]) for i in ds_train.indices)
val_subjects = set(str(ds_val._samples[int(i)]) for i in ds_val.indices)
overlap = train_subjects & val_subjects
assert len(overlap) == 0, f'Subject overlap: {overlap}'
print(f'Train: {len(ds_train)} frames from {len(train_subjects)} subjects, Val: {len(ds_val)} frames from {len(val_subjects)} subjects, Subject overlap: 0 ✓')
"
```

- [ ] **Step 2: Verify deterministic few-shot sampling**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from data.memmap_dataset import MemmapDataset
ds1 = MemmapDataset('data/mmfi_pose', split='all', envs=['env2'], seed=42,
                     few_shot_frames=5, few_shot_subjects=4)
ds2 = MemmapDataset('data/mmfi_pose', split='all', envs=['env2'], seed=42,
                     few_shot_frames=5, few_shot_subjects=4)
assert list(ds1.indices) == list(ds2.indices), 'Indices differ'
subject_ids = sorted(set(str(ds1._samples[int(i)]) for i in ds1.indices))
print(f'{len(ds1.indices)} few-shot frames from {len(subject_ids)} subjects: {subject_ids} ✓')
"
```

- [ ] **Step 3: Verify few-shot loader train/val disjoint**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from dataloader import create_few_shot_data_loader
r = create_few_shot_data_loader('data/mmfi_pose', envs=['env2'],
    batch_size=4, few_shot_frames=5, few_shot_subjects=4, seed=42)
train_idx = set(int(i) for i in r['train_indices'])
val_idx = set(int(i) for i in r['val'].dataset.indices)
overlap = train_idx & val_idx
assert len(overlap) == 0, f'Overlap: {overlap}'
print(f'Train: {len(r[\"train\"].dataset)} frames, Val: {len(r[\"val\"].dataset)} frames, Overlap: 0 ✓')
"
```

- [ ] **Step 4: Verify source_only mode runs without build_targets errors**

```bash
python train.py --dataset-root data/mmfi_pose --mode source_only \
    --source-envs env1 --epochs 1 --batch-size 4 --output-dir /tmp/test_cleanup
```

Expected: No `TypeError: unexpected keyword argument 'build_targets'` error.

- [ ] **Step 5: Verify finetune mode saves indices file**

```bash
python train.py --dataset-root data/mmfi_pose --mode finetune \
    --finetune-from /tmp/test_cleanup/best_val_mpjpe.pth \
    --target-envs env2 --few-shot-frames 5 --few-shot-subjects 4 \
    --freeze-tier 1 --epochs 1 --batch-size 4 \
    --output-dir /tmp/test_finetune_cleanup
ls -la /tmp/test_finetune_cleanup/few_shot_train_indices.npy
```

Expected: File exists, non-zero size.

- [ ] **Step 6: Verify eval.py --eval-split and --exclude-indices work**

```bash
python eval.py --dataset-root data/mmfi_pose \
    --checkpoint /tmp/test_finetune_cleanup/last.pth \
    --eval-split all --eval-envs env2 \
    --exclude-indices /tmp/test_finetune_cleanup/few_shot_train_indices.npy \
    --output-dir /tmp/test_eval_cleanup
```

Expected: Prints exclusion count, runs evaluation without error.

- [ ] **Step 7: Commit any fixes discovered during verification**

