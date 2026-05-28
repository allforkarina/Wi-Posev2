# Few-Shot Domain Adaptation + Tier 1 Fine-Tuning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable three training modes: (1) source-only supervised, (2) baseline zero-finetune eval on target, (3) Tier 1 fine-tune on target few-shot labels with selective param freezing.

**Architecture:** Extend `train.py` with a `--mode` flag (`source_only` | `da` | `finetune`), add few-shot frame sampling to `MemmapDataset`, and implement BN/LN/queries/coord_head selective freezing. No new scripts — all modes live in the existing `train.py`.

**Tech Stack:** Python 3.10+, torch 2.x, numpy, existing WiFlow codebase.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `data/memmap_dataset.py` | Add few-shot frame/action/subject selection |
| `dataloader.py` | Add few-shot loader factory for target domain |
| `train.py` | Add `--mode`, `--finetune-from`, single-domain epoch, freeze/unfreeze logic |

---

### Task 1: Few-Shot Sampling in MemmapDataset

**Files:**
- Modify: `data/memmap_dataset.py`

- [ ] **Step 1: Add few-shot parameters to `__init__`**

```python
# Add to __init__ signature after `seed: int = 42`:
few_shot_frames: int = 0,
few_shot_subjects: int = 0,
```

- [ ] **Step 2: Store and apply few-shot filtering in `_build_split`**

After `indices = self._build_split(...)`, add few-shot filtering. Implement a helper `_sample_few_shot`:

```python
def _sample_few_shot(
    self,
    indices: np.ndarray,
    few_shot_frames: int,
    few_shot_subjects: int,
    seed: int,
) -> np.ndarray:
    """Select at most `few_shot_frames` per action×subject, and at most
    `few_shot_subjects` subjects total (randomly chosen)."""
    if few_shot_frames <= 0 and few_shot_subjects <= 0:
        return indices

    rng = random.Random(seed + 1)

    # Group by (action, subject)
    grouped: dict[tuple[str, str], list[int]] = {}
    for idx in indices:
        idx_int = int(idx)
        key = (str(self._actions[idx_int]), str(self._samples[idx_int]))
        grouped.setdefault(key, []).append(idx_int)

    # Limit subjects if specified
    if few_shot_subjects > 0:
        all_subjects = sorted(set(k[1] for k in grouped))
        chosen_subjects = set(rng.sample(all_subjects, min(few_shot_subjects, len(all_subjects))))
        grouped = {k: v for k, v in grouped.items() if k[1] in chosen_subjects}

    # Limit frames per action×subject
    result: list[int] = []
    for (action, subject), frame_indices in sorted(grouped.items()):
        if few_shot_frames > 0:
            sampled = rng.sample(frame_indices, min(few_shot_frames, len(frame_indices)))
        else:
            sampled = frame_indices
        result.extend(sampled)

    return np.asarray(sorted(result), dtype=np.int64)
```

- [ ] **Step 3: Wire the filter in `__init__`**

After `self.indices = self._build_split(...)`, add:

```python
self.indices = self._sample_few_shot(
    self.indices, few_shot_frames, few_shot_subjects, seed,
)
```

- [ ] **Step 4: Commit**

```bash
git add data/memmap_dataset.py
git commit -m "feat: add few-shot frame/subject sampling to MemmapDataset"
```

---

### Task 2: Few-Shot Dataloader Factory

**Files:**
- Modify: `dataloader.py`

- [ ] **Step 1: Add `create_few_shot_data_loader`**

```python
def create_few_shot_data_loader(
    data_dir: str | Path,
    envs: Sequence[str],
    batch_size: int,
    few_shot_frames: int = 5,
    few_shot_subjects: int = 4,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Create train/val DataLoaders with few-shot target-domain sampling.

    ``split="all"`` filtered to envs, then few-shot sampling reduces to
    ≤ few_shot_frames per action×subject and ≤ few_shot_subjects.
    The result is split by subject into train (1 - val_ratio) and val.
    """
    full_dataset = MemmapDataset(
        data_dir=data_dir,
        split="all",
        envs=list(envs),
        seed=seed,
        build_targets=False,
        few_shot_frames=few_shot_frames,
        few_shot_subjects=few_shot_subjects,
    )

    # Split remaining subjects into train/val
    subjects = sorted(set(
        str(full_dataset._samples[int(i)])
        for i in full_dataset.indices
    ))
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * val_ratio))
    val_subjects = set(subjects[:n_val])
    train_subjects = set(subjects[n_val:])

    train_dataset = MemmapDataset(
        data_dir=data_dir, split="all", envs=list(envs),
        seed=seed, build_targets=False,
        few_shot_frames=few_shot_frames,
        few_shot_subjects=few_shot_subjects,
    )
    train_dataset.indices = np.asarray(sorted(
        [i for i in train_dataset.indices
         if str(train_dataset._samples[int(i)]) in train_subjects]
    ), dtype=np.int64)

    val_dataset = MemmapDataset(
        data_dir=data_dir, split="all", envs=list(envs),
        seed=seed, build_targets=False,
        few_shot_frames=few_shot_frames,
        few_shot_subjects=few_shot_subjects,
    )
    val_dataset.indices = np.asarray(sorted(
        [i for i in val_dataset.indices
         if str(val_dataset._samples[int(i)]) in val_subjects]
    ), dtype=np.int64)

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
    return {"train": train_loader, "val": val_loader}
```

- [ ] **Step 2: Import `random` at the top of `dataloader.py`**

```python
import random
```

- [ ] **Step 3: Commit**

```bash
git add dataloader.py
git commit -m "feat: add create_few_shot_data_loader for target-domain few-shot sampling"
```

---

### Task 3: Single-Domain Training Mode

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add new fields to `TrainConfig`**

```python
# After val_every:
mode: str = "da"           # "source_only" | "da" | "finetune"
few_shot_frames: int = 0
few_shot_subjects: int = 0
finetune_from: str = ""    # path to source checkpoint
```

- [ ] **Step 2: Add single-domain training epoch function**

```python
def run_single_domain_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion_config: TrainConfig,
    device: torch.device,
    optimizer: AdamW | None = None,
    scheduler: LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    use_amp = scaler is not None

    totals: Dict[str, float] = {}
    sample_count = 0

    for batch in loader:
        model_input, target = prepare_model_input(batch, device)
        bs = target.shape[0]

        with torch.set_grad_enabled(is_training):
            with torch.amp.autocast(device.type, enabled=use_amp):
                prediction = model(model_input)
                losses = compute_losses(
                    prediction, target,
                    bone_loss_weight=criterion_config.bone_loss_weight,
                )

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(losses["loss"]).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=criterion_config.grad_clip_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=criterion_config.grad_clip_norm,
                    )
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        kp_pred = extract_prediction_keypoints(prediction).detach()
        metrics = compute_metrics(kp_pred, target)

        sample_count += bs
        for name, value in {**losses, **metrics}.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * bs

    return average_meter_totals(totals, sample_count)
```

- [ ] **Step 3: Add `--mode`, `--few-shot-frames`, `--few-shot-subjects` CLI args**

```python
parser.add_argument("--mode", default="da", choices=["source_only", "da", "finetune"],
                    help="Training mode.")
parser.add_argument("--few-shot-frames", type=int, default=0,
                    help="Max frames per action×subject in target domain (0=all).")
parser.add_argument("--few-shot-subjects", type=int, default=0,
                    help="Max subjects in target domain few-shot sampling (0=all).")
```

- [ ] **Step 4: Add `source_only` loader creation in `run_training`**

In `run_training`, before the main training loop, add a branch. When `config.mode == "source_only"`, create loaders filtered to source envs:

```python
from data.memmap_dataset import MemmapDataset

if config.mode == "source_only":
    train_dataset = MemmapDataset(
        data_dir=config.dataset_root, split="train",
        envs=list(config.source_envs), seed=config.seed,
        build_targets=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, collate_fn=memmap_collate_fn,
        pin_memory=True, persistent_workers=config.num_workers > 0,
    )
    val_dataset = MemmapDataset(
        data_dir=config.dataset_root, split="val",
        envs=list(config.source_envs), seed=config.seed,
        build_targets=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, collate_fn=memmap_collate_fn,
        pin_memory=True, persistent_workers=config.num_workers > 0,
    )
```

- [ ] **Step 5: Wire source_only loop in `run_training`**

```python
if config.mode == "source_only":
    model = WiFlowModel(
        input_channels=3, axial_mode=config.axial_mode,
        decoder_type=config.decoder_type, heatmap_size=config.heatmap_size,
        dropout=config.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=config.max_lr, epochs=config.epochs,
                           steps_per_epoch=len(train_loader), pct_start=config.pct_start,
                           anneal_strategy="cos", div_factor=config.max_lr / max(config.lr, 1e-8),
                           final_div_factor=1000.0)
    scaler = torch.amp.GradScaler(device.type, enabled=config.amp and device.type == "cuda")

    for epoch in range(1, config.epochs + 1):
        # Same loop body as finetune (Step 3 of Task 4):
        # run_single_domain_epoch → do_val check → val → checkpoint → log → early stop
```

- [ ] **Step 6: Commit**

```bash
git add train.py
git commit -m "feat: add source_only training mode and single-domain epoch"
```

---

### Task 4: Finetune Mode + Tier 1 Parameter Freezing

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add freeze/unfreeze utility function**

```python
def apply_finetune_tier(
    model: nn.Module,
    tier: int = 1,
) -> None:
    """Freeze parameters according to tier level.

    Tier 0: freeze nothing (all trainable).
    Tier 1: freeze Conv + MHA + FFN weights. Keep BN/LN affine,
            joint_queries, and coordinate_head trainable.
    """
    if tier == 0:
        return

    # Tier 1: freeze everything first, then selectively unfreeze
    for p in model.parameters():
        p.requires_grad = False

    # Modules to keep trainable
    KEEP_PATTERNS = [
        "norm",          # LayerNorm
        "attention_norm",# LayerNorm in decoder attention
        "joint_queries", # learnable query vectors
        "coordinate_head",# decoder output MLP
    ]

    for name, module in model.named_modules():
        keep = any(pat in name for pat in KEEP_PATTERNS)
        if keep:
            for p in module.parameters():
                p.requires_grad = True

    # Also keep all BatchNorm affine params in spatial_encoder
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d,)):
            for p in module.parameters():
                p.requires_grad = True

    # Print summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Finetune Tier {tier}: {trainable:,} / {total:,} params trainable "
          f"({100 * trainable / total:.1f}%)")
```

- [ ] **Step 2: Add `--finetune-from` and `--freeze-tier` CLI args**

```python
parser.add_argument("--finetune-from", default="",
                    help="Path to source-only checkpoint for fine-tuning.")
parser.add_argument("--freeze-tier", type=int, default=1,
                    help="Parameter freezing tier (1 = BN+LN+queries+coord_head only).")
```

- [ ] **Step 3: Wire finetune mode in `run_training`**

```python
if config.mode == "finetune":
    # Load source checkpoint
    checkpoint = torch.load(config.finetune_from, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {config.finetune_from} "
          f"(epoch {checkpoint.get('epoch', '?')})")

    # Apply tier-based freezing
    apply_finetune_tier(model, tier=config.freeze_tier)

    # Use few-shot target data loaders
    from dataloader import create_few_shot_data_loader
    loaders = create_few_shot_data_loader(
        data_dir=config.dataset_root,
        envs=list(config.target_envs),
        batch_size=config.batch_size,
        few_shot_frames=config.few_shot_frames,
        few_shot_subjects=config.few_shot_subjects,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    print(f"Few-shot target: {len(train_loader.dataset)} train / "
          f"{len(val_loader.dataset)} val samples")

    # Train with single-domain loop (no CECE, no ICAL)
    for epoch in range(1, config.epochs + 1):
        start_time = time.perf_counter()
        do_val = epoch % config.val_every == 0 or epoch == config.epochs

        train_metrics = run_single_domain_epoch(
            model, train_loader, config, device,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - start_time

        if do_val:
            val_metrics = run_val_epoch(model, val_loader, config, device)
            save_checkpoint(output_dir / "last.pth", model, optimizer, scheduler,
                            epoch, best_metric=val_metrics["mpjpe"], config=config)
            if val_metrics["mpjpe"] < best_val_mpjpe:
                best_val_mpjpe = val_metrics["mpjpe"]
                patience_counter = 0
                save_checkpoint(output_dir / "best_val_mpjpe.pth", model, optimizer,
                                scheduler, epoch, best_metric=best_val_mpjpe, config=config)
            else:
                patience_counter += 1
            if val_metrics["pck_0_2"] > best_val_pck_0_2:
                best_val_pck_0_2 = val_metrics["pck_0_2"]
                save_checkpoint(output_dir / "best_val_pck_0_2.pth", model, optimizer,
                                scheduler, epoch, best_metric=best_val_pck_0_2, config=config)
        else:
            val_metrics = {}

        row = {
            "epoch": epoch, "train_loss": train_metrics["loss"],
            "train_coord_loss": train_metrics["coord_loss"],
            "train_bone_loss": train_metrics["bone_loss"],
            "train_mpjpe": train_metrics["mpjpe"],
            "train_pck_0_2": train_metrics["pck_0_2"],
            "val_mpjpe": val_metrics.get("mpjpe", ""),
            "val_pck_0_2": val_metrics.get("pck_0_2", ""),
            "current_lr": current_lr, "epoch_time": epoch_time,
        }
        append_csv_row(log_path, row)

        if do_val:
            print(f"epoch={epoch:03d} loss={train_metrics['loss']:.6f} "
                  f"val_mpjpe={val_metrics['mpjpe']:.6f} "
                  f"val_pck_0_2={val_metrics['pck_0_2']:.6f} "
                  f"lr={current_lr:.2e} epoch_time={epoch_time:.1f}s")
        else:
            print(f"epoch={epoch:03d} loss={train_metrics['loss']:.6f} "
                  f"(skip val) lr={current_lr:.2e} epoch_time={epoch_time:.1f}s")

        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

- [ ] **Step 4: Add `--finetune-lr` for lower LR during fine-tuning**

```python
parser.add_argument("--finetune-lr", type=float, default=1e-5,
                    help="Learning rate for fine-tuning (default: 1e-5).")

# In finetune mode, use finetune_lr instead of lr for the optimizer
ft_lr = config.finetune_lr if config.mode == "finetune" else config.lr
```

Add to TrainConfig:
```python
finetune_lr: float = 1e-5
```

- [ ] **Step 5: Commit**

```bash
git add train.py
git commit -m "feat: add finetune mode with Tier 1 selective parameter freezing"
```

---

### Task 5: Update CLAUDE.md with New Commands

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add Phase 1-3 commands**

```markdown
# Phase 1: Source-only training
python train.py --dataset-root data/mmfi_pose --mode source_only \
    --source-envs env1 --epochs 50 --batch-size 128 \
    --output-dir outputs/train_source

# Phase 2: Baseline eval (zero-finetune)
python eval.py --dataset-root data/mmfi_pose \
    --checkpoint outputs/train_source/best_val_mpjpe.pth \
    --output-dir outputs/eval_source_on_target

# Phase 3: Tier 1 fine-tune
python train.py --dataset-root data/mmfi_pose --mode finetune \
    --finetune-from outputs/train_source/best_val_mpjpe.pth \
    --target-envs env2 --few-shot-frames 5 --few-shot-subjects 4 \
    --freeze-tier 1 --epochs 30 --batch-size 128 --finetune-lr 1e-5 \
    --output-dir outputs/finetune_tier1
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add few-shot fine-tuning workflow commands"
```

---

### Task 6: Verification

**Files:** None (manual verification)

- [ ] **Step 1: Verify source_only mode runs**

```bash
python train.py --dataset-root data/mmfi_pose --mode source_only \
    --source-envs env1 --epochs 1 --batch-size 4 --output-dir /tmp/test_source
```

Expected: "Sanity shapes" line, single training epoch without DA metrics.

- [ ] **Step 2: Verify finetune mode runs**

```bash
python train.py --dataset-root data/mmfi_pose --mode finetune \
    --finetune-from outputs/train_source/best_val_mpjpe.pth \
    --target-envs env2 --few-shot-frames 5 --few-shot-subjects 4 \
    --freeze-tier 1 --epochs 1 --batch-size 4 --output-dir /tmp/test_ft
```

Expected: "Loaded checkpoint from ..." + "Finetune Tier 1: ~40K / ~2M params trainable" + training run.

- [ ] **Step 3: Verify freeze/unfreeze percentages**

Run a quick Python check:
```python
import sys; sys.path.insert(0, '.')
from models import WiFlowModel
from train import apply_finetune_tier
m = WiFlowModel()
apply_finetune_tier(m, tier=1)
total = sum(p.numel() for p in m.parameters())
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
assert 0.01 < trainable/total < 0.05, f"Expected ~2%, got {trainable/total:.2%}"
print(f"Tier 1: {trainable}/{total} = {trainable/total:.2%} ✓")
```

- [ ] **Step 4: Commit any fixes discovered during verification**
