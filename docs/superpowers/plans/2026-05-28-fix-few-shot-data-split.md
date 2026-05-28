# Fix Few-Shot Data Split & Eval Filtering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make few-shot sampling deterministic (sorted subjects, uniform frame spacing), rewrite val set to be all remaining target-domain data, save train indices for eval exclusion, and add env/split/exclude filtering to eval.py.

**Architecture:** Modify `MemmapDataset._sample_few_shot` for deterministic selection, rewrite `create_few_shot_data_loader` so val = complement of few-shot train in target domain, save `few_shot_train_indices.npy` during finetune, and extend `eval.py` with `--eval-split`, `--eval-envs`, `--exclude-indices` args.

**Tech Stack:** Python 3.10+, numpy, torch, existing MemmapDataset/DataLoader infra.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `data/memmap_dataset.py` | Deterministic few-shot sampling (sorted subjects, uniform frames) |
| `dataloader.py` | Rewrite `create_few_shot_data_loader`, extend `create_memmap_data_loader` for `all` split |
| `eval.py` | Add `--eval-split`, `--exclude-indices` args, wire loader filtering |
| `train.py` | Save `few_shot_train_indices.npy` in finetune mode |
| `CLAUDE.md` | Update Phase 2/3 commands |

---

### Task 1: Deterministic Few-Shot Sampling

**Files:**
- Modify: `data/memmap_dataset.py:128-161`

- [ ] **Step 1: Rewrite `_sample_few_shot` with deterministic selection**

Replace the random-based sampling with sorted-subject + uniform-frame selection:

```python
def _sample_few_shot(
    self,
    indices: np.ndarray,
    few_shot_frames: int,
    few_shot_subjects: int,
    seed: int,
) -> np.ndarray:
    """Select at most ``few_shot_subjects`` subjects (first by sorted ID)
    and at most ``few_shot_frames`` frames per action×subject via uniform
    spacing (linspace over sorted frame indices)."""
    if few_shot_frames <= 0 and few_shot_subjects <= 0:
        return indices

    # Group by (action, subject)
    grouped: dict[tuple[str, str], list[int]] = {}
    for idx in indices:
        idx_int = int(idx)
        key = (str(self._actions[idx_int]), str(self._samples[idx_int]))
        grouped.setdefault(key, []).append(idx_int)

    # Subject selection: sort by ID, take first K
    if few_shot_subjects > 0:
        all_subjects = sorted(set(k[1] for k in grouped))
        chosen_subjects = set(all_subjects[:few_shot_subjects])
        grouped = {k: v for k, v in grouped.items() if k[1] in chosen_subjects}

    # Frame selection: sort indices, uniform spacing via linspace
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

- [ ] **Step 2: Commit**

```bash
git add data/memmap_dataset.py
git commit -m "refactor: deterministic few-shot sampling (sorted subjects, uniform frames)"
```

---

### Task 2: Rewrite `create_few_shot_data_loader`

**Files:**
- Modify: `dataloader.py:144-201`
- Modify: `dataloader.py:16` (SPLIT_NAMES)

- [ ] **Step 1: Add `"all"` to SPLIT_NAMES**

```python
SPLIT_NAMES = ("train", "val", "test", "all")
```

- [ ] **Step 2: Rewrite `create_few_shot_data_loader`**

New signature returns `{"train": DataLoader, "val": DataLoader, "train_indices": np.ndarray}`.

Val = all target-domain data (env-filtered, `split="all"`) EXCEPT few-shot train indices.

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
    ≤ few_shot_frames per action×subject, deterministic uniform spacing).

    Val: ALL data in *envs* excluding the few-shot train indices.
    This is the full complement — every labelled frame in the target
    domain that was NOT used for few-shot training.

    Returns dict with keys: ``"train"``, ``"val"``, ``"train_indices"``.
    """
    train_dataset = MemmapDataset(
        data_dir=data_dir, split="all", envs=list(envs),
        seed=seed, build_targets=False,
        few_shot_frames=few_shot_frames,
        few_shot_subjects=few_shot_subjects,
    )
    train_indices = train_dataset.indices.copy()

    full_dataset = MemmapDataset(
        data_dir=data_dir, split="all", envs=list(envs),
        seed=seed, build_targets=False,
    )
    train_idx_set = set(int(i) for i in train_indices)
    val_indices = np.asarray(sorted(
        [i for i in full_dataset.indices if int(i) not in train_idx_set]
    ), dtype=np.int64)
    val_dataset = MemmapDataset(
        data_dir=data_dir, split="all", envs=list(envs),
        seed=seed, build_targets=False,
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

Remove the now-unused `val_ratio` parameter and the `_make_dataset` helper. Remove `import random` if no longer needed (check: `create_few_shot_data_loader` no longer uses `random.Random`, but `create_few_shot_data_loader` in the file still uses it — actually wait, the old version used `rng = random.Random(seed)` to shuffle subjects. The new version doesn't shuffle, so we can drop `random` if nothing else uses it).

Check: `random` is also imported in `dataloader.py:6`. Let me check what else uses it... Only `create_few_shot_data_loader` used `random.Random`. So after the rewrite, the `import random` at line 6 can be removed.

- [ ] **Step 3: Remove unused `import random` from dataloader.py**

```python
# Delete line 6: import random
```

- [ ] **Step 4: Commit**

```bash
git add dataloader.py
git commit -m "refactor: rewrite few-shot loader — val = complement of train in target domain"
```

---

### Task 3: Extend `eval.py` with `--eval-split` and `--exclude-indices`

**Files:**
- Modify: `eval.py:282-285` (CLI args)
- Modify: `eval.py:289-301` (main loader creation)

- [ ] **Step 1: Add `--exclude-indices` CLI arg**

```python
parser.add_argument(
    "--exclude-indices", default=None,
    help="Path to .npy file containing frame indices to exclude from evaluation.",
)
```

Add it after the `--eval-split` argument (already added at line 282-285).

- [ ] **Step 2: Wire `--eval-split` and `--exclude-indices` in `main()`**

Replace the hardcoded `split="test"` with `args.eval_split`, and add exclusion logic:

```python
test_loader = create_memmap_data_loader(
    data_dir=args.dataset_root,
    split=args.eval_split,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False,
    envs=args.eval_envs,
)

if args.exclude_indices is not None:
    exclude_path = Path(args.exclude_indices)
    if not exclude_path.exists():
        raise FileNotFoundError(f"Exclude indices file not found: {exclude_path}")
    exclude_set = set(np.load(str(exclude_path)).astype(int).tolist())
    dataset = test_loader.dataset
    dataset.indices = np.asarray(sorted(
        [i for i in dataset.indices if int(i) not in exclude_set]
    ), dtype=np.int64)
    print(f"Excluded {len(exclude_set)} indices, "
          f"{len(dataset.indices)} samples remaining for evaluation.")
```

Add `import numpy as np` at the top of eval.py.

- [ ] **Step 3: Commit**

```bash
git add eval.py
git commit -m "feat: add --eval-split and --exclude-indices to eval.py"
```

---

### Task 4: Save Train Indices in Finetune Mode

**Files:**
- Modify: `train.py:571-600` (`_setup_finetune`)

- [ ] **Step 1: Save `few_shot_train_indices.npy` in `_setup_finetune`**

The `_setup_finetune` function currently receives `config` and `device`, but doesn't have `output_dir`. We need to either:
1. Pass `output_dir` to `_setup_finetune`
2. Save the indices in `run_training` after calling `_setup_finetune`

Option 2 is simpler — have `_setup_finetune` return `train_indices` alongside the model and loaders, then save in `run_training`.

Update `_setup_finetune` return type and the return statement:

```python
def _setup_finetune(
    config: TrainConfig, device: torch.device,
) -> tuple[WiFlowModel, DataLoader, DataLoader, np.ndarray]:
    ...
    train_indices = loaders["train_indices"]
    print(f"Few-shot target: {len(loaders['train'].dataset)} train / "
          f"{len(loaders['val'].dataset)} val samples")
    return model, loaders["train"], loaders["val"], train_indices
```

In `run_training`, update the finetune branch (around line 622-628):

```python
elif config.mode == "finetune":
    model, train_loader, val_loader, fs_indices = _setup_finetune(config, device)
    train_loader_for_epoch = train_loader
    val_loader_for_epoch = val_loader
    da_mode = False
    steps_per_epoch = len(train_loader)
    lr = config.finetune_lr
    # Save few-shot train indices for later eval exclusion
    np.save(str(output_dir / "few_shot_train_indices.npy"), fs_indices)
    print(f"Saved few-shot train indices ({len(fs_indices)} frames) to "
          f"{output_dir / 'few_shot_train_indices.npy'}")
```

Add `import numpy as np` at the top of `train.py` (check: it's not currently imported; numpy is used via torch).

- [ ] **Step 2: Commit**

```bash
git add train.py
git commit -m "feat: save few-shot train indices in finetune mode for eval exclusion"
```

---

### Task 5: Update CLAUDE.md Commands

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Phase 2 and Phase 3 commands**

Phase 2 — baseline eval on target domain with `--eval-split all --eval-envs env2`:

```markdown
# Phase 2: Baseline eval (source checkpoint on ALL target domain data)
python eval.py --dataset-root data/mmfi_pose \
    --checkpoint outputs/train_source/best_val_mpjpe.pth \
    --eval-split all --eval-envs env2 \
    --output-dir outputs/eval_baseline
```

Phase 3 eval — evaluate finetuned model excluding few-shot training samples:

```markdown
# Phase 3: Tier 1 fine-tune
python train.py --dataset-root data/mmfi_pose --mode finetune \
    --finetune-from outputs/train_source/best_val_mpjpe.pth \
    --target-envs env2 --few-shot-frames 5 --few-shot-subjects 4 \
    --freeze-tier 1 --epochs 30 --batch-size 128 --finetune-lr 1e-5 \
    --output-dir outputs/finetune_tier1

# Phase 3 eval: evaluate finetuned model excluding few-shot train samples
python eval.py --dataset-root data/mmfi_pose \
    --checkpoint outputs/finetune_tier1/best_val_mpjpe.pth \
    --eval-split all --eval-envs env2 \
    --exclude-indices outputs/finetune_tier1/few_shot_train_indices.npy \
    --output-dir outputs/eval_finetune_tier1
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update Phase 2/3 eval commands with new eval flags"
```

---

### Task 6: Verification

**Files:** None (manual verification)

- [ ] **Step 1: Verify deterministic few-shot sampling**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from data.memmap_dataset import MemmapDataset
ds1 = MemmapDataset('data/mmfi_pose', split='all', envs=['env2'], seed=42,
                     few_shot_frames=5, few_shot_subjects=4, build_targets=False)
ds2 = MemmapDataset('data/mmfi_pose', split='all', envs=['env2'], seed=42,
                     few_shot_frames=5, few_shot_subjects=4, build_targets=False)
assert list(ds1.indices) == list(ds2.indices), 'Indices differ across runs'
print(f'Deterministic: {len(ds1.indices)} few-shot indices reproducible ✓')
"
```

- [ ] **Step 2: Verify few-shot loader returns complement val**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from dataloader import create_few_shot_data_loader
result = create_few_shot_data_loader('data/mmfi_pose', envs=['env2'],
    batch_size=4, few_shot_frames=5, few_shot_subjects=4, seed=42)
train_idx = set(int(i) for i in result['train_indices'])
val_idx = set(int(i) for i in result['val'].dataset.indices)
overlap = train_idx & val_idx
assert len(overlap) == 0, f'Train/val overlap: {overlap}'
print(f'Train: {len(result[\"train\"].dataset)}, Val: {len(result[\"val\"].dataset)}, Overlap: 0 ✓')
"
```

- [ ] **Step 3: Verify eval.py --eval-split and --exclude-indices parse correctly**

```bash
python -c "
from eval import parse_args
import sys
sys.argv = ['eval.py', '--dataset-root', 'data/mmfi_pose', '--checkpoint', 'dummy.pth',
            '--eval-split', 'all', '--eval-envs', 'env2',
            '--exclude-indices', '/tmp/test.npy']
args = parse_args()
assert args.eval_split == 'all'
assert args.eval_envs == ['env2']
assert args.exclude_indices == '/tmp/test.npy'
print('CLI args parse correctly ✓')
"
```

- [ ] **Step 4: Commit any fixes discovered during verification**

