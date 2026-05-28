# Single-Domain Regression + Cross-Domain Finetune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip heatmap code, add `source_only`/`finetune` training modes with few-shot cross-domain finetuning and Tier 1 freeze.

**Architecture:** Two-phase approach — first delete/clean all heatmap code across 7 files, then add few-shot sampling + finetune support across 5 files. Source-only mode preserves existing single-domain training behavior exactly. Finetune mode does deterministic few-shot sampling on target env, freezes ~98.5% of params, trains with no per-epoch val.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, pathlib

---

### Task 1: Delete heatmap-only files

**Files:**
- Delete: `models/wiflow_heatmap_decoder.py`
- Delete: `scripts/diagnose_loss.py`
- Delete: `docs/models_architecture_analysis.md`

- [ ] **Step 1: Delete the three files**

```bash
rm models/wiflow_heatmap_decoder.py
rm scripts/diagnose_loss.py
rm docs/models_architecture_analysis.md
```

- [ ] **Step 2: Commit**

```bash
git add -A && git commit -m "chore: delete heatmap-only files (heatmap_decoder, diagnose_loss, old arch doc)"
```

---

### Task 2: Clean `data/heatmap_gt.py` — remove PCM/PAF, keep generic utilities

**Files:**
- Modify: `data/heatmap_gt.py`

- [ ] **Step 1: Read current file to confirm content**

```bash
wc -l data/heatmap_gt.py
```

- [ ] **Step 2: Delete `OPENPOSE_18_NAMES` (lines 8-27)**

```python
# Delete from line 8 to line 27 (inclusive):
# OPENPOSE_18_NAMES = [...]  (entire list)
```

- [ ] **Step 3: Delete `LIMBS_18` (lines 49-69)**

```python
# Delete from line 49 to line 69 (inclusive):
# LIMBS_18 = [...]  (entire list)
```

- [ ] **Step 4: Delete `pose_to_heatmap_coords` function (lines 106-120)**

```python
# Delete the entire function
```

- [ ] **Step 5: Delete `heatmap_to_pose_coords` function (lines 123-133)**

```python
# Delete the entire function
```

- [ ] **Step 6: Delete `gaussian_2d` function (lines 136-141)**

```python
# Delete the entire function
```

- [ ] **Step 7: Delete `paf_line` function (lines 144-170)**

```python
# Delete the entire function
```

- [ ] **Step 8: Delete `build_pcm_paf` function (lines 173-198)**

```python
# Delete the entire function
```

- [ ] **Step 9: Verify remaining content: `COCO17_TO_OPENPOSE18`, `valid_point`, `coco17_to_openpose18` only**

```bash
grep "^def " data/heatmap_gt.py
```
Expected output:
```
def valid_point(point: np.ndarray) -> bool:
def coco17_to_openpose18(kpts17: np.ndarray) -> np.ndarray:
```

- [ ] **Step 10: Update module docstring** — Replace the entire file's opening lines 1-6 to reflect new purpose:

```python
from __future__ import annotations

from typing import Iterable

import numpy as np
```

- [ ] **Step 11: Commit**

```bash
git add data/heatmap_gt.py && git commit -m "refactor: remove PCM/PAF from heatmap_gt, keep coco17_to_openpose18 and valid_point"
```

---

### Task 3: Clean `pose_targets.py` — remove all heatmap functions

**Files:**
- Modify: `pose_targets.py`

- [ ] **Step 1: Delete `keypoints_to_heatmap_coords` (lines 8-15)**

- [ ] **Step 2: Delete `build_pcm_targets` (lines 18-36)**

- [ ] **Step 3: Delete `build_paf_targets` (lines 39-82)**

- [ ] **Step 4: Delete `build_pcm_paf_targets` (lines 85-94)**

- [ ] **Step 5: Delete `decode_pcm_argmax` (lines 97-106)**

- [ ] **Step 6: Verify file is now empty (or remove the `from models.skeleton import...` import and leave empty file for future use)**

Final content should be:
```python
from __future__ import annotations
```

- [ ] **Step 7: Commit**

```bash
git add pose_targets.py && git commit -m "refactor: remove all heatmap target construction from pose_targets.py"
```

---

### Task 4: Clean `models/__init__.py` — remove heatmap decoder exports

**Files:**
- Modify: `models/__init__.py`

- [ ] **Step 1: Replace entire file content**

```python
from .skeleton import NUM_OPENPOSE_KEYPOINTS, OPENPOSE_BONE_EDGES, build_normalized_adjacency
from .wiflow_axial_encoder import AXIAL_ENCODER_MODES, WiFlowAxialEncoder
from .wiflow_hierarchical_joint_decoder import WiFlowHierarchicalJointDecoder
from .wiflow_joint_decoder import WiFlowJointDecoder
from .wiflow_model import DECODER_TYPES, WiFlowModel
from .wiflow_spatial_encoder import WiFlowSpatialEncoder

__all__ = [
    "WiFlowModel",
    "WiFlowSpatialEncoder",
    "WiFlowAxialEncoder",
    "AXIAL_ENCODER_MODES",
    "DECODER_TYPES",
    "WiFlowJointDecoder",
    "WiFlowHierarchicalJointDecoder",
    "OPENPOSE_BONE_EDGES",
    "NUM_OPENPOSE_KEYPOINTS",
    "build_normalized_adjacency",
]
```

- [ ] **Step 2: Commit**

```bash
git add models/__init__.py && git commit -m "refactor: remove heatmap decoder exports from models/__init__"
```

---

### Task 5: Clean `models/wiflow_model.py` — remove heatmap_msfn support

**Files:**
- Modify: `models/wiflow_model.py`

- [ ] **Step 1: Remove `heatmap_decoder` and `pose_targets` imports (lines 7, 11)**

Delete:
```python
from .wiflow_heatmap_decoder import WiFlowMSFNDecoder
from pose_targets import decode_pcm_argmax
```

- [ ] **Step 2: Remove `heatmap_msfn` from `DECODER_TYPES` (line 13)**

Replace:
```python
DECODER_TYPES = ("joint", "hierarchical", "heatmap_msfn")
```
With:
```python
DECODER_TYPES = ("joint", "hierarchical")
```

- [ ] **Step 3: Remove `heatmap_size` parameter from `__init__` (lines 24, 32, 40)**

Replace the `__init__` signature:
```python
    def __init__(
        self,
        input_channels: int = 3,
        axial_mode: str = "spatial_then_temporal",
        decoder_type: str = "joint",
        heatmap_size: int = 36,
    ) -> None:
```
With:
```python
    def __init__(
        self,
        input_channels: int = 3,
        axial_mode: str = "spatial_then_temporal",
        decoder_type: str = "joint",
    ) -> None:
```

Remove line 32: `self.heatmap_size = heatmap_size`

Replace the decoder selection (lines 35-40):
```python
        if decoder_type == "joint":
            self.decoder = WiFlowJointDecoder()
        elif decoder_type == "hierarchical":
            self.decoder = WiFlowHierarchicalJointDecoder()
        else:
            self.decoder = WiFlowMSFNDecoder(heatmap_size=heatmap_size)
```
With:
```python
        if decoder_type == "joint":
            self.decoder = WiFlowJointDecoder()
        elif decoder_type == "hierarchical":
            self.decoder = WiFlowHierarchicalJointDecoder()
```

- [ ] **Step 4: Simplify `decode_features` method (lines 42-48)**

Replace:
```python
    def decode_features(self, x: torch.Tensor):
        decoder_output = self.decoder(x)
        if self.decoder_type != "heatmap_msfn":
            return decoder_output
        stages = decoder_output
        keypoints = decode_pcm_argmax(stages[-1]["pcm"])
        return {"keypoints": keypoints, "stages": stages}
```
With:
```python
    def decode_features(self, x: torch.Tensor):
        return self.decoder(x)
```

- [ ] **Step 5: Commit**

```bash
git add models/wiflow_model.py && git commit -m "refactor: remove heatmap_msfn decoder support from WiFlowModel"
```

---

### Task 6: Clean `train.py` — remove heatmap-related code

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Remove `build_pcm_paf_targets` import (line 19)**

Delete:
```python
from pose_targets import build_pcm_paf_targets
```

- [ ] **Step 2: Remove heatmap config fields from `TrainConfig` (lines 40-43)**

Delete:
```python
    heatmap_size: int = 36
    heatmap_sigma: float = 1.5
    paf_width: float = 1.0
    paf_loss_weight: float = 1.0
```

- [ ] **Step 3: Simplify `extract_prediction_keypoints` (lines 76-84)**

Remove the heatmap decoder branch. Replace the entire function with:
```python
def extract_prediction_keypoints(prediction: torch.Tensor) -> torch.Tensor:
    if not isinstance(prediction, torch.Tensor):
        raise TypeError(f"Unexpected model prediction type: {type(prediction)!r}")
    return prediction
```

- [ ] **Step 4: Simplify `compute_losses` (lines 87-130)**

Remove the heatmap branch and all heatmap parameters. Replace the entire function with:
```python
def compute_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    bone_loss_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    coord = F.l1_loss(prediction, target)
    bone = bone_length_loss(prediction, target)
    total = coord + bone_loss_weight * bone
    return {
        "loss": total,
        "coord_loss": coord,
        "bone_loss": bone,
    }
```

- [ ] **Step 5: Update call site in `run_epoch` (lines 184-192)**

Replace:
```python
            losses = compute_losses(
                prediction,
                target,
                bone_loss_weight=criterion_config.bone_loss_weight,
                heatmap_size=criterion_config.heatmap_size,
                heatmap_sigma=criterion_config.heatmap_sigma,
                paf_width=criterion_config.paf_width,
                paf_loss_weight=criterion_config.paf_loss_weight,
            )
```
With:
```python
            losses = compute_losses(
                prediction,
                target,
                bone_loss_weight=criterion_config.bone_loss_weight,
            )
```

- [ ] **Step 6: Remove heatmap config from model instantiation in `run_training` (lines 280-285)**

Replace:
```python
    model = WiFlowModel(
        input_channels=3,
        axial_mode=config.axial_mode,
        decoder_type=config.decoder_type,
        heatmap_size=config.heatmap_size,
    ).to(device)
```
With:
```python
    model = WiFlowModel(
        input_channels=3,
        axial_mode=config.axial_mode,
        decoder_type=config.decoder_type,
    ).to(device)
```

- [ ] **Step 7: Remove heatmap columns from CSV row (lines 329-351)**

Replace the `row` dict with:
```python
        row: Dict[str, float | int | str] = {
            "epoch": epoch,
            "axial_mode": config.axial_mode,
            "decoder_type": config.decoder_type,
            "train_loss": train_metrics["loss"],
            "train_coord_loss": train_metrics["coord_loss"],
            "train_bone_loss": train_metrics["bone_loss"],
            "train_mpjpe": train_metrics["mpjpe"],
            "train_pck_0_2": train_metrics["pck_0_2"],
            "val_loss": val_metrics["loss"],
            "val_coord_loss": val_metrics["coord_loss"],
            "val_bone_loss": val_metrics["bone_loss"],
            "val_mpjpe": val_metrics["mpjpe"],
            "val_pck_0_2": val_metrics["pck_0_2"],
            "val_pck_0_5": val_metrics["pck_0_5"],
            "current_lr": current_lr,
            "epoch_time": epoch_time,
        }
```

- [ ] **Step 8: Remove `DECODER_TYPES` heatmap_msfn from `parse_args` (lines 403-408)**

No code change needed in `parse_args` — `DECODER_TYPES` is imported and will only have `("joint", "hierarchical")` after Task 5.

- [ ] **Step 9: Commit**

```bash
git add train.py && git commit -m "refactor: remove heatmap config and loss branches from train.py"
```

---

### Task 7: Clean `evaluation/feature_viz.py` — remove PCM/PAF figures

**Files:**
- Modify: `evaluation/feature_viz.py`

- [ ] **Step 1: Delete `_fig5a_pcm_radar` function (lines 636-681)**

Delete from `# Figure 5a: PCM Peak Response Radar` header through the blank line after `_save_fig(...)` at line 681.

- [ ] **Step 2: Delete `_fig5b_paf_direction_consistency` function (lines 684-761)**

Delete from `# Figure 5b: PAF Direction Consistency` header through the blank line after `_save_fig(...)` at line 761.

- [ ] **Step 3: Update fig name list (lines 927-939)**

Replace:
```python
    if decoder_type in ("joint", "hierarchical"):
        fig_names.append("fig4_joint_query_trajectory")
        fig_names.append("fig6_feature_pose_correlation")
    else:
        fig_names.append("fig5a_pcm_radar")
        fig_names.append("fig5b_paf_direction")
        fig_names.append("fig6_feature_pose_correlation")
```
With:
```python
    fig_names.append("fig4_joint_query_trajectory")
    fig_names.append("fig6_feature_pose_correlation")
```

- [ ] **Step 4: Update docstring (line 1130)**

Replace:
```python
        One of ``"joint"``, ``"hierarchical"``, ``"heatmap_msfn"``.
```
With:
```python
        One of ``"joint"``, ``"hierarchical"``.
```

- [ ] **Step 5: Update decoder_hooks branching (lines 1184-1185)**

Replace:
```python
    else:
        decoder_hooks = ["decoder"]
```
With — remove the `else` branch since `decoder_type` can only be `"joint"` or `"hierarchical"`. Add validation:

```python
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}")
```

- [ ] **Step 6: Remove heatmap figure call (lines 1217-1220)**

Delete:
```python
            # Fig 5: Heatmap Quality (heatmap_msfn only)
            if decoder_type == "heatmap_msfn":
                _fig5a_pcm_radar(sample, ctx, sample_dir)
                _fig5b_paf_direction_consistency(sample, ctx, sample_dir)
```

- [ ] **Step 7: Commit**

```bash
git add evaluation/feature_viz.py && git commit -m "refactor: remove PCM/PAF figure generation from feature_viz"
```

---

### Task 8: Add `"all"` split + `_sample_few_shot` to `data/memmap_dataset.py`

**Files:**
- Modify: `data/memmap_dataset.py`

- [ ] **Step 1: Remove heatmap imports and parameters from `__init__`**

Remove line 11:
```python
from data.heatmap_gt import build_pcm_paf
```

Remove parameters from `__init__` signature (lines 43-47):
```python
        heatmap_size: int = 36,
        heatmap_sigma: float = 1.5,
        paf_width: float = 1.0,
        build_targets: bool = True,
```

Remove attribute assignments (lines 53-57):
```python
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.paf_width = paf_width
        self.build_targets = build_targets
```

- [ ] **Step 2: Remove PCM/PAF building in `__getitem__` (lines 142-151)**

Delete:
```python
        if self.build_targets:
            pcm, paf = build_pcm_paf(
                kpts18,
                size=self.heatmap_size,
                sigma=self.heatmap_sigma,
                paf_width=self.paf_width,
                pose_range=self.pose_range,
            )
            item["pcm"] = torch.from_numpy(pcm)
            item["paf"] = torch.from_numpy(paf)
```

- [ ] **Step 3: Update `__init__` signature to remove dead params (lines 40-41)**

Remove `time_packets` and `subcarrier_mode` and `heatmap_size/sigma/paf_width/pose_range/build_targets` from signature:

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
    ) -> None:
```

Update `__init__` body to remove corresponding attribute assignments. Keep only:
```python
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"split must be train/val/test/all, got {split}")
        self.split = split
        self.normalize = normalize

        data_dir = Path(data_dir)
        if normalize not in CSI_FILES:
            raise ValueError(f"Unknown normalize mode: {normalize}, expected one of {list(CSI_FILES)}")

        self._csi = np.load(str(data_dir / CSI_FILES[normalize]), mmap_mode="r")
        self._kpts18 = np.load(str(data_dir / "ground_truth.npy"))
        meta = np.load(str(data_dir / "meta.npz"), allow_pickle=True)
        self._envs = meta["environment"]
        self._samples = meta["sample"]
        self._actions = meta["action"]

        self.indices = self._build_split(
            split, envs, train_subjects, test_subjects, random_val_ratio, seed
        )
```

- [ ] **Step 4: Add `_sample_few_shot` method after `_build_split`**

Insert after the `_build_split` method:

```python
    def _sample_few_shot(
        self,
        few_shot_subjects: int,
        few_shot_frames: int,
    ) -> list[int]:
        action_list = [str(a) for a in self._actions]
        sample_list = [str(s) for s in self._samples]

        unique_subjects = sorted(set(sample_list))
        selected_subjects = unique_subjects[:few_shot_subjects]
        selected_set = set(selected_subjects)

        selected_indices: list[int] = []
        for subject in selected_subjects:
            for action in sorted(set(action_list)):
                group = [
                    i for i in range(len(self._actions))
                    if sample_list[i] == subject and action_list[i] == action
                ]
                if not group:
                    continue
                group.sort()
                if len(group) <= few_shot_frames:
                    selected_indices.extend(group)
                else:
                    sampled = np.linspace(0, len(group) - 1, few_shot_frames, dtype=int)
                    selected_indices.extend([group[s] for s in sampled])

        return sorted(selected_indices)
```

- [ ] **Step 5: Commit**

```bash
git add data/memmap_dataset.py && git commit -m "feat: add _sample_few_shot deterministic sampling, remove heatmap params from MemmapDataset"
```

---

### Task 9: Update `dataloader.py` — add `create_few_shot_data_loader`, envs/``all`` support

**Files:**
- Modify: `dataloader.py`

- [ ] **Step 1: Replace SPLIT_NAMES with ALL_SPLITS**

Replace line 14:
```python
SPLIT_NAMES = ("train", "val", "test")
```
With:
```python
SPLIT_NAMES = ("train", "val", "test")
ALL_SPLITS = ("train", "val", "test", "all")
```

- [ ] **Step 2: Update `create_memmap_data_loader` to accept `envs` and `split="all"`**

Replace the entire function (lines 31-57):
```python
def create_memmap_data_loader(
    data_dir: str | Path,
    split: str,
    batch_size: int,
    envs: tuple[str, ...] | None = None,
    num_workers: int = 0,
    shuffle: bool | None = None,
    seed: int = 42,
) -> DataLoader:
    if split not in ALL_SPLITS:
        raise ValueError(f"split must be one of {ALL_SPLITS}, got {split}")

    dataset = MemmapDataset(
        data_dir=data_dir,
        split=split,
        envs=envs,
        seed=seed,
    )
    should_shuffle = shuffle if shuffle is not None else split in ("train", "all")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        collate_fn=memmap_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
```

- [ ] **Step 3: Add `create_few_shot_data_loader` function after `create_memmap_data_loaders`**

Insert after the existing `create_memmap_data_loaders` function:
```python
def create_few_shot_data_loader(
    data_dir: str | Path,
    target_envs: tuple[str, ...],
    few_shot_subjects: int,
    few_shot_frames: int,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[int]]:
    full_dataset = MemmapDataset(
        data_dir=data_dir,
        split="all",
        envs=target_envs,
        seed=seed,
    )
    train_indices = full_dataset._sample_few_shot(
        few_shot_subjects=few_shot_subjects,
        few_shot_frames=few_shot_frames,
    )
    train_dataset = Subset(full_dataset, train_indices)
    all_indices = list(range(len(full_dataset)))
    val_indices = [i for i in all_indices if i not in set(train_indices)]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=memmap_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_dataset = Subset(full_dataset, val_indices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=memmap_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_indices
```

- [ ] **Step 4: Add `Subset` import at top**

Add to imports (line 10):
```python
from torch.utils.data import DataLoader, Subset
```

- [ ] **Step 5: Remove `create_memmap_data_loaders` debug `main` and `parse_args` (lines 78-102)**

Delete the entire debug main block (lines 78 through end of file):
```python
def parse_args() -> argparse.Namespace:
    ...
def main() -> None:
    ...
if __name__ == "__main__":
    main()
```

Also remove `argparse` import (line 5):
```python
import argparse
```

- [ ] **Step 6: Commit**

```bash
git add dataloader.py && git commit -m "feat: add create_few_shot_data_loader, envs filter, 'all' split support"
```

---

### Task 10: Update `train.py` — add `--mode`, few-shot args, Tier 1 freeze, finetune loop

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add `mode`, `source_envs`, `target_envs`, finetune fields to `TrainConfig`**

Add after `seed`:
```python
    # Mode / env selection
    mode: str  # "source_only" or "finetune"
    source_envs: tuple[str, ...] | None = None
    target_envs: tuple[str, ...] | None = None
    # Finetune
    finetune_from: str | None = None
    few_shot_subjects: int = 4
    few_shot_frames: int = 5
    freeze_tier: int = 1
```

- [ ] **Step 2: Add `apply_finetune_tier` function**

Insert before `run_training`:
```python
def apply_finetune_tier(model: nn.Module, tier: int = 1) -> int:
    trainable_params = 0
    if tier == 1:
        for name, param in model.named_parameters():
            keep = any(kw in name.lower() for kw in ("norm", "bn", "ln", "joint_queries", "coordinate_head"))
            param.requires_grad = keep
            if keep:
                trainable_params += param.numel()
    else:
        raise ValueError(f"Unknown freeze tier: {tier}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Freeze tier {tier}: {trainable_params}/{total} parameters trainable ({trainable_params / total * 100:.1f}%)")
    return trainable_params
```

- [ ] **Step 3: Add `run_finetune_epoch` function**

Insert after `run_epoch`:
```python
def run_finetune_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    criterion_config: TrainConfig,
    device: torch.device,
    optimizer: AdamW,
    scheduler: LRScheduler | None = None,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    sample_count = 0

    for batch in loader:
        model_input, target = prepare_model_input(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(model_input)
        losses = compute_losses(
            prediction,
            target,
            bone_loss_weight=criterion_config.bone_loss_weight,
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=criterion_config.grad_clip_norm,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = target.shape[0]
        sample_count += batch_size
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size

    return average_meter_totals(totals, sample_count)
```

- [ ] **Step 4: Rewrite `run_training` to dispatch by `config.mode`**

Replace the entire `run_training` function:

```python
def run_training(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    output_dir = Path(config.output_dir)

    if config.mode == "source_only":
        _run_source_only(config, device, output_dir)
    elif config.mode == "finetune":
        _run_finetune(config, device, output_dir)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")
```

- [ ] **Step 5: Rename existing `run_training` body to `_run_source_only`**

Move the existing source-only training loop into `_run_source_only`:
```python
def _run_source_only(config: TrainConfig, device: torch.device, output_dir: Path) -> None:
    loaders = create_memmap_data_loaders(
        data_dir=config.dataset_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    # ... (rest of existing code, unchanged)
```

In `_run_source_only`, the `create_memmap_data_loaders` call needs `envs` support. Replace:
```python
    loaders = create_memmap_data_loaders(
        data_dir=config.dataset_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
```
With:
```python
    loaders = {}
    envs = config.source_envs if config.source_envs else None
    for split in ("train", "val", "test"):
        loaders[split] = create_memmap_data_loader(
            data_dir=config.dataset_root,
            split=split,
            batch_size=config.batch_size,
            envs=envs if split in ("val", "test") else envs,
            num_workers=config.num_workers,
            seed=config.seed,
        )
```

- [ ] **Step 6: Add `_run_finetune` function**

```python
def _run_finetune(config: TrainConfig, device: torch.device, output_dir: Path) -> None:
    from dataloader import create_few_shot_data_loader

    target_envs = config.target_envs
    if not target_envs:
        raise ValueError("--target-envs required for finetune mode")

    train_loader, val_loader, train_indices = create_few_shot_data_loader(
        data_dir=config.dataset_root,
        target_envs=target_envs,
        few_shot_subjects=config.few_shot_subjects,
        few_shot_frames=config.few_shot_frames,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    print(f"Few-shot train: {len(train_indices)} frames, val: {len(val_loader.dataset)} frames")

    indices_path = output_dir / "few_shot_train_indices.npy"
    indices_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(indices_path), np.array(train_indices, dtype=np.int64))

    if not config.finetune_from:
        raise ValueError("--finetune-from required for finetune mode")
    checkpoint = torch.load(config.finetune_from, map_location=device)
    model = WiFlowModel(
        input_channels=3,
        axial_mode=config.axial_mode,
        decoder_type=config.decoder_type,
    ).to(device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    apply_finetune_tier(model, tier=config.freeze_tier)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.max_lr,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=config.max_lr / max(config.lr, 1e-8),
        final_div_factor=1000.0,
    )

    first_batch = next(iter(train_loader))
    model_input, target = prepare_model_input(first_batch, device)
    with torch.no_grad():
        output = model(model_input)
    print(
        "Sanity shapes: "
        f"input={tuple(model_input.shape)}, output={tuple(output.shape)}, label={tuple(target.shape)}"
    )

    best_train_loss = float("inf")
    log_path = output_dir / "train_log.csv"
    for epoch in range(1, config.epochs + 1):
        start_time = time.perf_counter()
        train_metrics = run_finetune_epoch(
            model, train_loader, config, device, optimizer, scheduler
        )
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - start_time

        row: Dict[str, float | int | str] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_coord_loss": train_metrics["coord_loss"],
            "train_bone_loss": train_metrics["bone_loss"],
            "current_lr": current_lr,
            "epoch_time": epoch_time,
        }
        append_csv_row(log_path, row)

        save_checkpoint(
            output_dir / f"epoch_{epoch:03d}.pth",
            model, optimizer, scheduler, epoch,
            best_metric=train_metrics["loss"],
            config=config,
        )
        if train_metrics["loss"] < best_train_loss:
            best_train_loss = train_metrics["loss"]
            save_checkpoint(
                output_dir / "best_train_loss.pth",
                model, optimizer, scheduler, epoch,
                best_metric=best_train_loss,
                config=config,
            )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"best_train_loss={best_train_loss:.6f} "
            f"lr={current_lr:.2e} "
            f"epoch_time={epoch_time:.1f}s"
        )
```

- [ ] **Step 7: Update `parse_args`**

Replace the entire `parse_args` function:
```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the WiFlow pose model.")
    parser.add_argument("--mode", required=True, choices=("source_only", "finetune"))
    parser.add_argument("--dataset-root", required=True, help="Path to the NPY memmap dataset directory.")
    parser.add_argument("--output-dir", default="outputs/train", help="Directory for logs and checkpoints.")
    parser.add_argument("--axial-mode", default="spatial_then_temporal", choices=AXIAL_ENCODER_MODES)
    parser.add_argument("--decoder-type", default="joint", choices=DECODER_TYPES)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--source-envs", nargs="*", default=None, help="Source environment names (for source_only)")
    parser.add_argument("--target-envs", nargs="*", default=None, help="Target environment names (for finetune)")
    parser.add_argument("--finetune-from", default=None, help="Path to source checkpoint for finetune")
    parser.add_argument("--few-shot-subjects", type=int, default=4)
    parser.add_argument("--few-shot-frames", type=int, default=5)
    parser.add_argument("--freeze-tier", type=int, default=1)
    return parser.parse_args()
```

- [ ] **Step 8: Update `main`**

```python
def main() -> None:
    args = parse_args()
    config_dict = vars(args)
    config_dict["source_envs"] = tuple(config_dict["source_envs"]) if config_dict["source_envs"] else None
    config_dict["target_envs"] = tuple(config_dict["target_envs"]) if config_dict["target_envs"] else None
    config = TrainConfig(**{k: v for k, v in config_dict.items() if k in TrainConfig.__dataclass_fields__})
    run_training(config)
```

- [ ] **Step 9: Add `np` import**

Add at top:
```python
import numpy as np
```

- [ ] **Step 10: Commit**

```bash
git add train.py && git commit -m "feat: add source_only/finetune modes, Tier 1 freeze, few-shot finetune loop"
```

---

### Task 11: Update `eval.py` — add `--eval-envs`/`--exclude-indices`, remove `--eval-split`/heatmap_size

**Files:**
- Modify: `eval.py`

- [ ] **Step 1: Remove `heatmap_size` from `load_checkpoint_model` (line 50)**

Delete line 50:
```python
        heatmap_size=int(train_config.get("heatmap_size", 36)),
```

Update model construction:
```python
    model = WiFlowModel(
        input_channels=3,
        axial_mode=str(train_config.get("axial_mode", "spatial_then_temporal")),
        decoder_type=str(train_config.get("decoder_type", "joint")),
    ).to(device)
```

- [ ] **Step 2: Add new CLI arguments to `parse_args`**

Add after `--num-workers`:
```python
    parser.add_argument(
        "--eval-envs", nargs="*", default=None,
        help="Filter by environment names (e.g., --eval-envs env1 env2). Evaluates all if not set.",
    )
    parser.add_argument(
        "--exclude-indices", default=None,
        help="Path to .npy file containing frame indices to exclude from evaluation.",
    )
```

- [ ] **Step 3: Update `main` to use `--eval-envs` and `--exclude-indices`**

Replace the test loader construction (lines 274-280):
```python
    test_loader = create_memmap_data_loader(
        data_dir=args.dataset_root,
        split="test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
```
With:
```python
    from dataloader import create_memmap_data_loader as _create_loader

    eval_envs = tuple(args.eval_envs) if args.eval_envs else None
    test_dataset = MemmapDataset(
        data_dir=args.dataset_root,
        split="all",
        envs=eval_envs,
    )

    if args.exclude_indices:
        exclude = np.load(args.exclude_indices)
        exclude_set = set(exclude.tolist())
        keep = [i for i in range(len(test_dataset)) if i not in exclude_set]
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, keep)
        print(f"Excluded {len(exclude_set)} few-shot indices, {len(test_dataset)} remaining")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=memmap_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
```

- [ ] **Step 4: Add needed imports at top**

Add:
```python
import numpy as np

from data.memmap_dataset import MemmapDataset
from dataloader import memmap_collate_fn
```

- [ ] **Step 5: Commit**

```bash
git add eval.py && git commit -m "feat: add --eval-envs and --exclude-indices to eval.py, remove heatmap_size"
```

---

### Task 12: Update `AGENTS.md`

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Replace heatmap ablation commands**

Remove:
```
Run a MultiFormer-style MSFN heatmap decoder ablation:
...
python train.py --dataset-root data\mmfi_pose --decoder-type heatmap_msfn ...
```

- [ ] **Step 2: Replace default training command**

Replace:
```
python train.py --dataset-root data\mmfi_pose --epochs 50 --batch-size 64 --output-dir outputs\train
```
With:
```
python train.py --mode source_only --dataset-root data\mmfi_pose --epochs 50 --batch-size 64 --output-dir outputs\train
```

- [ ] **Step 3: Add finetune workflow section**

Add after ablation commands:
```
### Cross-Domain Few-Shot Finetune Pipeline

```powershell
# Phase 1: Source-only Training
python train.py --mode source_only --dataset-root data\mmfi_pose_v4 --source-envs env1 --output-dir outputs\source_baseline --epochs 50

# Phase 2: Baseline Evaluation
python eval.py --dataset-root data\mmfi_pose_v4 --checkpoint outputs\source_baseline\best_val_mpjpe.pth --eval-envs env2 --output-dir outputs\baseline_eval

# Phase 3: Few-shot Finetune
python train.py --mode finetune --dataset-root data\mmfi_pose_v4 --target-envs env2 --output-dir outputs\finetune --finetune-from outputs\source_baseline\best_val_mpjpe.pth --few-shot-subjects 4 --few-shot-frames 5 --epochs 30

# Phase 4: Post-FT Evaluation
python eval.py --dataset-root data\mmfi_pose_v4 --checkpoint outputs\finetune\best_train_loss.pth --eval-envs env2 --output-dir outputs\finetune_eval --exclude-indices outputs\finetune\few_shot_train_indices.npy
```
```

- [ ] **Step 4: Update decoder types**

Replace:
```
Supported `--decoder-type` values are `joint`, `hierarchical`, and `heatmap_msfn`.
```
With:
```
Supported `--decoder-type` values are `joint` and `hierarchical`.
```

- [ ] **Step 5: Remove heatmap-related references**

Remove all mentions of `heatmap_size`, `heatmap_sigma`, `paf_width`, `paf_loss_weight`, `pcm`, `paf`, `PCM`, `PAF`, `MSFN`, `build_pcm_paf` from the document.

- [ ] **Step 6: Update `pose_targets.py` and `data/heatmap_gt.py` description**

Replace:
```
- `data/heatmap_gt.py`: Functions for generating OpenPose18 PCM/PAF targets from normalized keypoint coordinates.
- `pose_targets.py`: Torch utilities for online OpenPose18 PCM/PAF target synthesis from normalized coordinates and argmax PCM decoding back to normalized keypoints.
```
With:
```
- `data/heatmap_gt.py`: OpenPose18 coordinate conversion utilities (coco17_to_openpose18, valid_point).
- `pose_targets.py`: Reserved for future pose target utilities.
```

- [ ] **Step 7: Commit**

```bash
git add AGENTS.md && git commit -m "docs: update AGENTS.md for single-domain + finetune workflow"
```

---

### Task 13: Final verification — run sanity check

- [ ] **Step 1: Run import check**

```bash
python -c "from models import WiFlowModel; m = WiFlowModel(); print('Model OK:', sum(p.numel() for p in m.parameters()))"
```
Expected: model instantiates without errors.

- [ ] **Step 2: Run CLI help for train.py**

```bash
python train.py --help
```
Expected: shows `--mode {source_only, finetune} (required)`.

- [ ] **Step 3: Run CLI help for eval.py**

```bash
python eval.py --help
```
Expected: shows `--eval-envs` and `--exclude-indices`, no `--eval-split`.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore: final verification — all imports clean, CLI help correct"
git push
```