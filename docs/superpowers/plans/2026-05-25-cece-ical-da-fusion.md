# CECE + ICAL DA Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fuse AdaPose's CECE (cross-environment channel enhancement) and ICAL (instance-level consistency alignment loss) into WiFlow as a module-decoupled domain adaptation training pipeline.

**Architecture:** Add `CECEModule` (stateless channel reweighting) in `models/cece.py`, `compute_ical_loss` and `run_da_epoch`/`run_val_epoch` in `train.py`, and `create_da_data_loaders` in `dataloader.py`. WiFlowModel's `forward()` is unchanged — DA logic is orchestrated entirely in the training loop. Source domain uses `split="all"`, target domain split into train/val/test.

**Tech Stack:** Python 3.10+, PyTorch, existing WiFlow codebase (no new dependencies).

---

## File Change Map

| Operation | File | Responsibility |
|-----------|------|---------------|
| Create | `models/cece.py` | `CECEModule` — stateless channel reweighting |
| Modify | `models/__init__.py` | Add `CECEModule` to exports |
| Modify | `dataloader.py` | Add `create_da_data_loaders` factory |
| Modify | `train.py` | `TrainConfig` DA fields, `compute_ical_loss`, `run_da_epoch`, `run_val_epoch`, `run_training` rewrite, CLI |

No other files changed.

---

### Task 1: Create CECEModule

**Files:**
- Create: `models/cece.py`

- [ ] **Step 1: Write the module file**

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CECEModule(nn.Module):
    """Cross-Environment Channel Enhancement.

    Computes per-channel cosine similarity between source and target domain
    feature maps, then reweights both domains' features by channel consistency
    scores.  Stateless — no learnable parameters.  Only used during training.
    """

    def __init__(self, num_channels: int = 256) -> None:
        super().__init__()
        self.num_channels = num_channels

    def forward(
        self,
        src_feat: torch.Tensor,
        tgt_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # src_feat, tgt_feat: [B, C, H, W]
        C = src_feat.shape[1]

        # Batch-average to obtain domain-level representative feature maps
        src_mean = src_feat.mean(dim=0)          # [C, H, W]
        tgt_mean = tgt_feat.mean(dim=0)          # [C, H, W]

        # Flatten spatial dims: each channel becomes a vector in R^{H*W}
        src_flat = src_mean.view(C, -1)          # [C, H*W]
        tgt_flat = tgt_mean.view(C, -1)          # [C, H*W]

        # Per-channel cosine similarity → [C], range [-1, 1]
        cos_sim = F.cosine_similarity(src_flat, tgt_flat, dim=1)

        # Linear map to [0, 1]; channels with negative similarity get weight < 0.5
        weights = (cos_sim + 1.0) / 2.0          # [C]
        weights = weights.view(1, C, 1, 1)       # broadcast shape

        return src_feat * weights, tgt_feat * weights
```

- [ ] **Step 2: Verify the module loads without errors**

Run: `python -c "from models.cece import CECEModule; m = CECEModule(); print('CECEModule created:', m)"`

Expected: `CECEModule created: CECEModule()`

- [ ] **Step 3: Quick shape test**

Run: `python -c "import torch; from models.cece import CECEModule; m = CECEModule(256); s = torch.randn(4,256,16,29); t = torch.randn(4,256,16,29); so, to_ = m(s, t); print('src out:', tuple(so.shape), 'tgt out:', tuple(to_.shape))"`

Expected: `src out: (4, 256, 16, 29) tgt out: (4, 256, 16, 29)`

- [ ] **Step 4: Commit**

```bash
git add models/cece.py
git commit -m "feat: add CECEModule for cross-environment channel enhancement"
```

---

### Task 2: Export CECEModule from models/__init__.py

**Files:**
- Modify: `models/__init__.py`

- [ ] **Step 1: Add import and export**

Edit `models/__init__.py` — add line after the existing wiflow_spatial_encoder import:

```diff
 from .skeleton import NUM_OPENPOSE_KEYPOINTS, OPENPOSE_BONE_EDGES, build_normalized_adjacency
+from .cece import CECEModule
 from .wiflow_axial_encoder import AXIAL_ENCODER_MODES, WiFlowAxialEncoder
```

And add `"CECEModule"` to `__all__`:

```diff
 __all__ = [
     "WiFlowModel",
     "WiFlowSpatialEncoder",
     "WiFlowAxialEncoder",
+    "CECEModule",
     "AXIAL_ENCODER_MODES",
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from models import CECEModule; print('import OK:', CECEModule)"`

Expected: `import OK: <class 'models.cece.CECEModule'>`

- [ ] **Step 3: Commit**

```bash
git add models/__init__.py
git commit -m "feat: export CECEModule from models package"
```

---

### Task 3: Add create_da_data_loaders to dataloader.py

**Files:**
- Modify: `dataloader.py`

- [ ] **Step 1: Add the factory function**

Add `from typing import Sequence` to the imports. Then add the following function after `create_memmap_data_loaders` (before `parse_args`):

```python
def create_da_data_loaders(
    data_dir: str | Path,
    source_envs: Sequence[str],
    target_envs: Sequence[str],
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Create DataLoaders for domain-adaptation training.

    Source domain uses all filtered data (``split="all"``, no val split).
    Target domain is split into train / val / test by subject (80/20).

    Returns a dict with keys:
      ``"source_train"``, ``"target_train"``, ``"target_val"``, ``"target_test"``.
    """
    source_dataset = MemmapDataset(
        data_dir=data_dir,
        split="all",
        envs=list(source_envs),
        seed=seed,
        build_targets=False,
    )
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=memmap_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    target_loaders: dict[str, DataLoader] = {}
    for split in SPLIT_NAMES:  # ("train", "val", "test")
        dataset = MemmapDataset(
            data_dir=data_dir,
            split=split,
            envs=list(target_envs),
            seed=seed,
            build_targets=False,
        )
        should_shuffle = split == "train"
        target_loaders[f"target_{split}"] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            num_workers=num_workers,
            collate_fn=memmap_collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    return {
        "source_train": source_loader,
        **target_loaders,
    }
```

- [ ] **Step 2: Update imports — add `Sequence`**

In the existing imports, change:
```python
from typing import Optional
```
to:
```python
from typing import Optional, Sequence
```

- [ ] **Step 3: Verify import and basic structure**

Run: `python -c "from dataloader import create_da_data_loaders; print('import OK')"`

Expected: `import OK`

- [ ] **Step 4: Commit**

```bash
git add dataloader.py
git commit -m "feat: add create_da_data_loaders for dual-domain data pipeline"
```

---

### Task 4: Add compute_ical_loss to train.py

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add the ICAL loss function**

Insert `compute_ical_loss` after the `compute_losses` function (after line 130). Add the import for `CECEModule` at the top:

In the imports section (line 17):
```diff
-from models import AXIAL_ENCODER_MODES, DECODER_TYPES, OPENPOSE_BONE_EDGES, WiFlowModel
+from models import AXIAL_ENCODER_MODES, CECEModule, DECODER_TYPES, OPENPOSE_BONE_EDGES, WiFlowModel
```

Insert after line 130 (after `compute_losses` ends):

```python
def compute_ical_loss(
    f_s: torch.Tensor,
    f_t: torch.Tensor,
    y_s_gt: torch.Tensor,
    y_t_pred: torch.Tensor,
    sigma_pose: float = 0.5,
) -> torch.Tensor:
    """Instance-level consistency alignment loss.

    Reweights feature-space distances by pose similarity so that
    source-target pairs with similar poses are aligned more strongly.

    Args:
        f_s:       Source features after CECE reweighting + GAP, shape [B, D].
        f_t:       Target features after CECE reweighting + GAP, shape [B, D].
        y_s_gt:    Source ground-truth keypoints, shape [B, 18, 2].
        y_t_pred:  Target predicted keypoints, shape [B, 18, 2].
        sigma_pose: Temperature for pose-distance → similarity mapping.

    Returns:
        Scalar ICAL loss.
    """
    y_s_flat = y_s_gt.flatten(1)                            # [B, 36]
    y_t_flat = y_t_pred.flatten(1)                          # [B, 36]

    # Pairwise pose distances
    pose_dist = torch.cdist(y_s_flat, y_t_flat)             # [B, B]

    # Pose similarity weights with row-wise normalisation
    weights = torch.exp(-pose_dist / sigma_pose)            # [B, B]
    weights = weights / weights.sum(dim=1, keepdim=True)    # row-normalised

    # Weighted squared L2 feature distances
    f_dist_sq = torch.cdist(f_s, f_t, p=2).pow(2)          # [B, B]

    # Divide by B to keep loss magnitude stable across batch sizes
    return (weights * f_dist_sq).sum() / f_s.shape[0]
```

- [ ] **Step 2: Verify the function imports and runs with dummy data**

Run:
```powershell
python -c "
import torch
from train import compute_ical_loss
f_s = torch.randn(4, 256)
f_t = torch.randn(4, 256)
y_s = torch.randn(4, 18, 2)
y_t = torch.randn(4, 18, 2)
loss = compute_ical_loss(f_s, f_t, y_s, y_t)
print('ICAL loss:', loss.item(), 'shape:', loss.shape)
"
```

Expected: `ICAL loss: <positive float> shape: torch.Size([])`

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add compute_ical_loss for instance-level cross-domain alignment"
```

---

### Task 5: Rewrite train.py — TrainConfig, run_da_epoch, run_val_epoch, run_training, CLI

**Files:**
- Modify: `train.py`

This task rewrites the core training infrastructure in `train.py`. The following existing functions are **preserved unchanged**: `prepare_model_input`, `bone_length_loss`, `extract_prediction_keypoints`, `compute_losses`, `compute_torso_scale`, `mpjpe`, `pck`, `compute_metrics`, `average_meter_totals`, `save_checkpoint`, `append_csv_row`, `select_device`.

The following are **replaced**: `run_epoch` → `run_da_epoch` + `run_val_epoch`, `run_training`, `parse_args`, `TrainConfig`.

The following are **removed**: `maybe_subset_loader`, `create_memmap_data_loaders` import.

- [ ] **Step 1: Update TrainConfig with DA fields**

Replace the existing `TrainConfig` dataclass (lines 27-47) with:

```python
@dataclass(frozen=True)
class TrainConfig:
    dataset_root: str
    output_dir: str = "outputs/train"
    axial_mode: str = "spatial_then_temporal"
    decoder_type: str = "joint"
    epochs: int = 50
    batch_size: int = 64
    lr: float = 2e-5
    max_lr: float = 5e-4
    weight_decay: float = 5e-4
    grad_clip_norm: float = 1.0
    bone_loss_weight: float = 0.5
    heatmap_size: int = 36
    heatmap_sigma: float = 1.5
    paf_width: float = 1.0
    paf_loss_weight: float = 1.0
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42
    subset_size: int | None = None
    # DA fields
    source_envs: tuple[str, ...] = ("lab",)
    target_envs: tuple[str, ...] = ("corridor",)
    alpha: float = 0.1
    ical_warmup_epochs: int = 5
    cece_enabled: bool = True
```

Note: `tuple[str, ...]` requires adding `Tuple` to the typing import if not already present. The `from __future__ import annotations` at the top of the file makes all annotations strings, so `tuple[str, ...]` works without importing `Tuple`.

- [ ] **Step 2: Update imports**

Replace:
```python
from dataloader import create_memmap_data_loaders
```
with:
```python
from dataloader import create_da_data_loaders
```

Remove the `Subset` import (no longer needed):
```python
from torch.utils.data import DataLoader, Subset
```
becomes:
```python
from torch.utils.data import DataLoader
```

Add `Tuple` to typing imports if needed — but with `from __future__ import annotations` already present, `tuple[str, ...]` works natively in Python 3.10+.

- [ ] **Step 3: Replace run_epoch with run_da_epoch**

Remove `run_epoch` (lines 166-212) and `maybe_subset_loader` (lines 248-257). Insert `run_da_epoch` in their place:

```python
def run_da_epoch(
    model: nn.Module,
    cece: CECEModule | None,
    source_loader: DataLoader,
    target_loader: DataLoader,
    criterion_config: TrainConfig,
    device: torch.device,
    epoch: int,
    optimizer: AdamW | None = None,
    scheduler: LRScheduler | None = None,
) -> Dict[str, float]:
    """Run one epoch of dual-domain training.

    Splits WiFlowModel's forward into spatial → CECE → decoder to
    compute per-domain supervised losses and the ICAL cross-domain loss.

    Source loader is iterated in a cycle; when exhausted the iterator
    is rebuilt to trigger a fresh shuffle (no fixed pairings across epochs).
    """
    is_training = optimizer is not None
    model.train(is_training)

    # ICAL warmup: linearly ramp alpha from 0 to config.alpha
    actual_alpha = criterion_config.alpha * min(
        1.0, epoch / max(criterion_config.ical_warmup_epochs, 1)
    )

    totals: Dict[str, float] = {}
    source_sample_count = 0
    target_sample_count = 0
    source_iter = iter(source_loader)

    for batch_t in target_loader:
        # --- source batch (cycle with re-shuffle) ---
        try:
            batch_s = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            batch_s = next(source_iter)

        x_s, kp_s_gt = prepare_model_input(batch_s, device)
        x_t, kp_t_gt = prepare_model_input(batch_t, device)

        bs_s = x_s.shape[0]
        bs_t = x_t.shape[0]

        with torch.set_grad_enabled(is_training):
            # Forward: spatial → axial
            feat_s = model.axial_encoder(model.spatial_encoder(x_s))
            feat_t = model.axial_encoder(model.spatial_encoder(x_t))

            # CECE channel reweighting
            if criterion_config.cece_enabled and cece is not None:
                feat_s_ce, feat_t_ce = cece(feat_s, feat_t)
            else:
                feat_s_ce, feat_t_ce = feat_s, feat_t

            # Decode
            y_s = model.decode_features(feat_s_ce)
            y_t = model.decode_features(feat_t_ce)

            # Supervised losses
            losses_s = compute_losses(
                y_s,
                kp_s_gt,
                bone_loss_weight=criterion_config.bone_loss_weight,
            )
            losses_t = compute_losses(
                y_t,
                kp_t_gt,
                bone_loss_weight=criterion_config.bone_loss_weight,
            )

            # ICAL loss
            f_s_pooled = feat_s_ce.mean(dim=[2, 3])           # GAP → [B, 256]
            f_t_pooled = feat_t_ce.mean(dim=[2, 3])           # GAP → [B, 256]
            y_s_keypoints = extract_prediction_keypoints(y_s)
            y_t_keypoints = extract_prediction_keypoints(y_t)
            loss_ical = compute_ical_loss(
                f_s_pooled, f_t_pooled, kp_s_gt, y_t_keypoints,
            )

            loss = (losses_s["loss"] + losses_t["loss"]) / 2.0 + actual_alpha * loss_ical

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=criterion_config.grad_clip_norm,
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        # Metrics
        kp_s_pred = extract_prediction_keypoints(y_s).detach()
        kp_t_pred = extract_prediction_keypoints(y_t).detach()
        metrics_s = compute_metrics(kp_s_pred, kp_s_gt)
        metrics_t = compute_metrics(kp_t_pred, kp_t_gt)

        source_sample_count += bs_s
        target_sample_count += bs_t

        source_metric_items = {
            "source_loss": losses_s["loss"],
            "source_coord_loss": losses_s["coord_loss"],
            "source_bone_loss": losses_s["bone_loss"],
            "source_mpjpe": metrics_s["mpjpe"],
            "source_pck_0_2": metrics_s["pck_0_2"],
        }
        target_metric_items = {
            "target_loss": losses_t["loss"],
            "target_coord_loss": losses_t["coord_loss"],
            "target_bone_loss": losses_t["bone_loss"],
            "target_mpjpe": metrics_t["mpjpe"],
            "target_pck_0_2": metrics_t["pck_0_2"],
        }
        for name, value in {**source_metric_items, **target_metric_items}.items():
            weight = bs_s if name.startswith("source") else bs_t
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * weight
        totals["ical"] = totals.get("ical", 0.0) + float(loss_ical.detach().cpu())

    # Average
    averaged: Dict[str, float] = {}
    for name, total in totals.items():
        if name == "ical":
            averaged[name] = total / max(step_count, 1)
        else:
            count = source_sample_count if name.startswith("source") else target_sample_count
            averaged[name] = total / max(count, 1)
    return averaged
```

- [ ] **Step 4: Add run_val_epoch (target-domain only validation)**

Insert after `run_da_epoch`:

```python
def run_val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion_config: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Run validation on the target domain only.

    Uses model.forward() directly — no CECE, no ICAL.
    Best checkpoint selection is based on ``val_mpjpe`` from this function.
    """
    model.eval()
    totals: Dict[str, float] = {}
    sample_count = 0

    with torch.no_grad():
        for batch in loader:
            model_input, target = prepare_model_input(batch, device)
            prediction = model(model_input)
            losses = compute_losses(
                prediction,
                target,
                bone_loss_weight=criterion_config.bone_loss_weight,
            )
            keypoint_prediction = extract_prediction_keypoints(prediction)
            metrics = compute_metrics(keypoint_prediction, target)

            bs = target.shape[0]
            sample_count += bs
            for name, value in {**losses, **metrics}.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * bs

    return average_meter_totals(totals, sample_count)
```

- [ ] **Step 5: Rewrite run_training**

Replace the existing `run_training` (lines 266-396) with:

```python
def run_training(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    output_dir = Path(config.output_dir)

    loaders = create_da_data_loaders(
        data_dir=config.dataset_root,
        source_envs=config.source_envs,
        target_envs=config.target_envs,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    source_train_loader = loaders["source_train"]
    target_train_loader = loaders["target_train"]
    target_val_loader = loaders["target_val"]

    model = WiFlowModel(
        input_channels=3,
        axial_mode=config.axial_mode,
        decoder_type=config.decoder_type,
        heatmap_size=config.heatmap_size,
    ).to(device)

    cece = CECEModule(num_channels=256).to(device) if config.cece_enabled else None

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.max_lr,
        epochs=config.epochs,
        steps_per_epoch=len(target_train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=config.max_lr / max(config.lr, 1e-8),
        final_div_factor=1000.0,
    )

    # Sanity check
    first_batch = next(iter(target_train_loader))
    model_input, target = prepare_model_input(first_batch, device)
    with torch.no_grad():
        output = model(model_input)
    keypoint_output = extract_prediction_keypoints(output)
    print(
        "Sanity shapes: "
        f"input={tuple(model_input.shape)}, output={tuple(keypoint_output.shape)}, label={tuple(target.shape)}"
    )
    if keypoint_output.shape != target.shape:
        raise ValueError(
            f"Model output shape {tuple(keypoint_output.shape)} does not match label shape {tuple(target.shape)}"
        )

    best_val_mpjpe = float("inf")
    best_val_pck_0_2 = -float("inf")
    log_path = output_dir / "train_log.csv"
    for epoch in range(1, config.epochs + 1):
        start_time = time.perf_counter()
        train_metrics = run_da_epoch(
            model=model,
            cece=cece,
            source_loader=source_train_loader,
            target_loader=target_train_loader,
            criterion_config=config,
            device=device,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        val_metrics = run_val_epoch(model, target_val_loader, config, device)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - start_time

        actual_alpha = config.alpha * min(
            1.0, epoch / max(config.ical_warmup_epochs, 1)
        )

        row: Dict[str, float | int | str] = {
            "epoch": epoch,
            "axial_mode": config.axial_mode,
            "decoder_type": config.decoder_type,
            "train_source_loss": train_metrics["source_loss"],
            "train_source_coord_loss": train_metrics["source_coord_loss"],
            "train_source_bone_loss": train_metrics["source_bone_loss"],
            "train_source_mpjpe": train_metrics["source_mpjpe"],
            "train_source_pck_0_2": train_metrics["source_pck_0_2"],
            "train_target_loss": train_metrics["target_loss"],
            "train_target_coord_loss": train_metrics["target_coord_loss"],
            "train_target_bone_loss": train_metrics["target_bone_loss"],
            "train_target_mpjpe": train_metrics["target_mpjpe"],
            "train_target_pck_0_2": train_metrics["target_pck_0_2"],
            "train_ical": train_metrics["ical"],
            "alpha": actual_alpha,
            "val_loss": val_metrics["loss"],
            "val_coord_loss": val_metrics["coord_loss"],
            "val_bone_loss": val_metrics["bone_loss"],
            "val_mpjpe": val_metrics["mpjpe"],
            "val_pck_0_2": val_metrics["pck_0_2"],
            "val_pck_0_5": val_metrics["pck_0_5"],
            "heatmap_size": config.heatmap_size,
            "heatmap_sigma": config.heatmap_sigma,
            "paf_width": config.paf_width,
            "paf_loss_weight": config.paf_loss_weight,
            "current_lr": current_lr,
            "epoch_time": epoch_time,
        }
        append_csv_row(log_path, row)

        save_checkpoint(
            output_dir / "last.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_metric=val_metrics["mpjpe"],
            config=config,
        )
        if val_metrics["mpjpe"] < best_val_mpjpe:
            best_val_mpjpe = val_metrics["mpjpe"]
            save_checkpoint(
                output_dir / "best_val_mpjpe.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric=best_val_mpjpe,
                config=config,
            )
        if val_metrics["pck_0_2"] > best_val_pck_0_2:
            best_val_pck_0_2 = val_metrics["pck_0_2"]
            save_checkpoint(
                output_dir / "best_val_pck_0_2.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric=best_val_pck_0_2,
                config=config,
            )

        print(
            f"epoch={epoch:03d} "
            f"src_loss={train_metrics['source_loss']:.6f} "
            f"tgt_loss={train_metrics['target_loss']:.6f} "
            f"ical={train_metrics['ical']:.6f} "
            f"val_mpjpe={val_metrics['mpjpe']:.6f} "
            f"val_pck_0_2={val_metrics['pck_0_2']:.6f} "
            f"lr={current_lr:.2e} "
            f"epoch_time={epoch_time:.1f}s"
        )
```

- [ ] **Step 6: Rewrite parse_args with DA CLI arguments**

Replace the existing `parse_args` (lines 399-408) with:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the WiFlow pose model with domain adaptation.")
    parser.add_argument("--dataset-root", required=True, help="Path to the NPY memmap dataset directory.")
    parser.add_argument("--output-dir", default="outputs/train", help="Directory for logs and checkpoints.")
    parser.add_argument("--axial-mode", default="spatial_then_temporal", choices=AXIAL_ENCODER_MODES)
    parser.add_argument("--decoder-type", default="joint", choices=DECODER_TYPES)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    # DA arguments
    parser.add_argument("--source-envs", nargs="+", default=["lab"],
                        help="Source domain environment names.")
    parser.add_argument("--target-envs", nargs="+", default=["corridor"],
                        help="Target domain environment names.")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="ICAL loss weight.")
    parser.add_argument("--ical-warmup-epochs", type=int, default=5,
                        help="Number of epochs to linearly ramp up ICAL alpha.")
    parser.add_argument("--no-cece", action="store_true", default=False,
                        help="Disable CECE channel reweighting.")
    return parser.parse_args()
```

- [ ] **Step 7: Rewrite main() to handle new CLI args**

Replace `main()` (lines 411-415) with:

```python
def main() -> None:
    args = parse_args()
    config_dict = vars(args)
    # Map CLI flags to config fields
    config_dict["cece_enabled"] = not config_dict.pop("no_cece")
    # Convert lists to tuples for frozen dataclass
    config_dict["source_envs"] = tuple(config_dict["source_envs"])
    config_dict["target_envs"] = tuple(config_dict["target_envs"])
    # Remove keys not in TrainConfig
    config = TrainConfig(**{
        k: v for k, v in config_dict.items()
        if k in TrainConfig.__dataclass_fields__
    })
    run_training(config)
```

- [ ] **Step 8: Verify the full module imports without errors**

Run: `python -c "import train; print('train.py imports OK')"`

Expected: `train.py imports OK`

- [ ] **Step 9: Dry-run CLI help**

Run: `python train.py --help`

Expected: Help text including `--source-envs`, `--target-envs`, `--alpha`, `--ical-warmup-epochs`, `--no-cece`.

- [ ] **Step 10: Commit**

```bash
git add train.py
git commit -m "feat: add dual-domain training loop with CECE + ICAL support"
```

---

### Task 6: Verify existing tests still pass

**Files:**
- (none modified — verification only)

- [ ] **Step 1: Run existing pytest suite**

```bash
conda activate WiFiPose && pytest tests/ -x -q --ignore=tests/test_wiflow_skeleton_decoder.py --ignore=tests/test_wiflow_temporal_encoder.py --ignore=tests/test_wiflow_spatial_temporal_fuser.py --ignore=tests/test_wiflow_attention_pooler.py
```

Expected: All non-legacy tests PASS. If any fail due to the `create_memmap_data_loaders` import removal from `train.py` (which is not imported by other modules), verify that no test depends on it.

- [ ] **Step 2: Fix any broken imports in tests**

If any test imports `run_epoch`, `maybe_subset_loader`, or `create_memmap_data_loaders` from `train.py`, those tests need updating. Check:

```bash
python -c "import ast, sys; [sys.stdout.write(f'{f}: {n.id}\\n') for f in ['tests/test_train.py', 'tests/test_eval.py'] for n in ast.walk(ast.parse(open(f).read())) if isinstance(n, ast.Name) and n.id in ('run_epoch', 'maybe_subset_loader', 'create_memmap_data_loaders')]"
```

If no output, no broken imports exist.

- [ ] **Step 3: Commit any test fixes if needed**

Only if Step 2 finds issues.

---

### Task 7: Final verification — training sanity check

- [ ] **Step 1: Run a 2-epoch sanity training**

```bash
python train.py --dataset-root data/mmfi_pose --source-envs lab --target-envs corridor --epochs 2 --batch-size 8 --output-dir outputs/sanity_da --subset-size 8
```

Expected:
- Sanity shapes print (input/output/label shapes match)
- Two epochs complete without errors
- `outputs/sanity_da/train_log.csv` exists with 2 rows
- `outputs/sanity_da/last.pth` checkpoint exists
- Loss values are finite (not NaN)

- [ ] **Step 2: Verify CSV has expected columns**

Run: `python -c "import csv; r=csv.DictReader(open('outputs/sanity_da/train_log.csv')); print(list(r.fieldnames))"`

Expected columns include: `train_source_loss`, `train_target_loss`, `train_ical`, `train_source_mpjpe`, `train_target_mpjpe`, `alpha`.

- [ ] **Step 3: Push to remote**

```bash
git push origin main
```