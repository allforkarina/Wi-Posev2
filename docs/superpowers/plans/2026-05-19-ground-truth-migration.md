# Ground Truth 数据源迁移与环境元数据管理 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将训练标签数据源从 `dataset/{action}/{subject}/rgb/` 迁移至 `/data/WiFiPose/dataset/ground_truth_npy/`，实现 `--env` 和 `--split-mode` 参数支持，重构 MemmapDataset 划分逻辑。

**Architecture:** 全量数据在 `build_memmap.py` 中拼接为 memmap .npy 文件，环境信息作为元数据保存。训练时通过 `--env` 筛选环境、通过 `--split-mode` 控制划分模式（subject 级别 7:2:1 或 frame 级别 7:2:1），参数从 train.py → dataloader.py → MemmapDataset 逐层透传。

**Tech Stack:** Python 3.10+, NumPy, PyTorch, scipy

---

### 设计决策汇总

| # | 决策 | 结论 |
|---|------|------|
| 1 | 环境-Subject 映射 | 一一映射，S01-S10→env1, ..., S31-S40→env4 |
| 2 | 划分策略 | `subject` 模式（S01-S07 train/S08-S09 val/S10 test）和 `frame` 模式（7:2:1 随机） |
| 3 | Ground truth 加载 | 整体加载 `[297, 17, 3]`，按帧索引取 `[17, 2]` |
| 4 | 坐标范围 | 已归一化到 `[-0.8, 0.8]`，不做缩放 |
| 5 | 关节点顺序 | COCO17 标准，映射表不变 |
| 6 | `build_memmap.py` | 全量拼接 4 环境，环境作为元数据 |
| 7 | 参数接口 | `--env env1`, `--split-mode subject|frame`，比例写死 |
| 8 | 参数透传 | 工厂函数追加两个参数，最小改动 |
| 9 | 有效性检查 | 移除 `_valid_point`、删除 `normalize_kpts_to_pose_range`，仅保留 NaN/Inf 防御 |
| 10 | 归一化统计量 | 基于全部样本（4 环境 × 40 subjects）计算，与训练时 `--env` 解耦 |

---

### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/build_memmap.py` | 修改 | 迁移 ground truth 数据源、简化坐标处理 |
| `data/memmap_dataset.py` | 修改 | 新增 `envs`/`split_mode` 参数、重写 `_build_split` |
| `dataloader.py` | 修改 | 工厂函数透传新参数 |
| `train.py` | 修改 | `TrainConfig` 新增字段、CLI 新增参数 |
| `tests/test_memmap_dataset.py` | 新建 | 数据集划分逻辑测试 |

---

### Task 1: 重构 `build_memmap.py` — 迁移 Ground Truth 数据源

**Files:**
- Modify: `scripts/build_memmap.py` (全文)

- [ ] **Step 1: 更新文档注释和 CLI 参数**

将文件顶部的文档字符串替换为新版描述，并在 `main()` 的 `argparse` 中新增 `--gt` 参数：

```python
"""
Build memory-mapped .npy files from MM-Fi dataset for fast training I/O.

Pre-computes 3 normalization variants and stores each as a single .npy file.
Training loader uses np.load(path, mmap_mode='r') for zero-copy OS-cached access.

Input:
    /data/WiFiPose/dataset/dataset/{ACTION}/{SUBJECT}/
        wifi-csi/frame*.mat   ← CSIamp (3, 114, 10) float64

    /data/WiFiPose/dataset/ground_truth_npy/
        Ezz_Syy_Axx.npy       ← COCO17 keypoints [297, 17, 3] float32

Output:
    /data/WiFiPose/dataset/mmfi_pose_v4/
        csi_gminmax.npy  ← global_minmax normalized (N, 64, 3, 114) float32
        csi_gzscore.npy  ← global_zscore normalized (N, 64, 3, 114) float32
        csi_zscore.npy   ← per-sample zscore normalized (N, 64, 3, 114) float32
        ground_truth.npy ← OpenPose18, pose_range (N, 18, 2) float32
        meta.npz         ← environment, sample, action, frame_idx
        stats.json       ← normalization statistics (computed on all samples)

Usage:
    python scripts/build_memmap.py \
        --src /data/WiFiPose/dataset/dataset \
        --gt /data/WiFiPose/dataset/ground_truth_npy \
        --dst /data/WiFiPose/dataset/mmfi_pose_v4 \
        --workers 8
"""
```

在 `main()` 的 `argparse` 中新增 `--gt` 参数：

```python
parser.add_argument("--gt", default="/data/WiFiPose/dataset/ground_truth_npy")
```

- [ ] **Step 2: 移除 `_valid_point` 和 `normalize_kpts_to_pose_range` 函数**

删除以下两个函数定义（约第95-111行）：

```python
# 删除 _valid_point
def _valid_point(point: np.ndarray) -> bool:
    point = np.asarray(point)
    return bool(np.isfinite(point).all() and not np.allclose(point, 0.0))

# 删除 normalize_kpts_to_pose_range
def normalize_kpts_to_pose_range(
    kpts: np.ndarray, pose_min: float = -0.8, pose_max: float = 0.8,
) -> np.ndarray:
    kpts = np.asarray(kpts, dtype=np.float32).copy()
    non_zero = kpts[kpts != 0]
    abs_max = float(np.abs(non_zero).max()) if len(non_zero) > 0 else 0.0
    if abs_max > 10.0:
        IMG_W, IMG_H = 1920.0, 1080.0
        kpts[..., 0] /= IMG_W
        kpts[..., 1] /= IMG_H
        span = pose_max - pose_min
        kpts = kpts * span + pose_min
    invalid = ~np.isfinite(kpts).all(axis=-1) | np.all(np.isclose(kpts, 0.0), axis=-1)
    kpts[invalid] = 0.0
    return kpts.astype(np.float32)
```

新增简化的坐标防御函数：

```python
def sanitize_kpts(kpts: np.ndarray) -> np.ndarray:
    kpts = np.asarray(kpts, dtype=np.float32)
    finite = np.isfinite(kpts)
    if finite.all():
        return kpts
    fill = float(np.median(kpts[finite])) if finite.any() else 0.0
    return np.nan_to_num(kpts, nan=fill, posinf=fill, neginf=fill).astype(np.float32)
```

- [ ] **Step 3: 更新 `iter_trials` — 移除对 `rgb/` 目录的依赖**

将 `iter_trials` 中检查 `rgb/` 的条件改为仅检查 `wifi-csi/`：

```python
def iter_trials(src_root: Path) -> list[Path]:
    trials: list[Path] = []
    for action_dir in sorted(p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("A")):
        for subj_dir in sorted(p for p in action_dir.iterdir() if p.is_dir() and p.name.startswith("S")):
            if (subj_dir / "wifi-csi").is_dir():
                trials.append(subj_dir)
    return trials
```

- [ ] **Step 4: 重写 `process_trial` — 使用 ground truth npy 文件**

将 `process_trial` 函数签名新增 `gt_dir` 参数，整体从 ground truth npy 文件加载标签：

```python
def process_trial(trial_dir: Path, gt_dir: Path) -> dict | None:
    action = trial_dir.parent.name
    subject = trial_dir.name
    env_num = (int(subject.lstrip("S")) - 1) // 10 + 1
    environment = f"env{env_num}"
    wifi_dir = trial_dir / "wifi-csi"

    mat_paths = sorted(wifi_dir.glob("frame*.mat"))
    if not mat_paths:
        return None

    gt_path = gt_dir / f"E{env_num:02d}_{subject}_{action}.npy"
    if not gt_path.is_file():
        return None
    gt_data = np.load(str(gt_path))  # [297, 17, 3]

    n_frames = len(mat_paths)
    csi_frames = np.empty((n_frames, TIME_PACKETS, RX_ANTENNAS, SUBCARRIERS), dtype=np.float32)
    kpts18 = np.zeros((n_frames, 18, 2), dtype=np.float32)
    frame_idx = np.zeros(n_frames, dtype=np.int64)

    for i, mat_path in enumerate(mat_paths):
        mat = sio.loadmat(str(mat_path))
        csi_frames[i] = preprocess_csi_one_frame(np.asarray(mat["CSIamp"], dtype=np.float32))
        frame_num_str = mat_path.stem.replace("frame", "")
        frame_num = int(frame_num_str) if frame_num_str else i
        kpts_coco17_xy = sanitize_kpts(gt_data[frame_num - 1, :, :2])  # [17, 2]
        kpts18[i] = coco17_to_openpose18(kpts_coco17_xy)
        frame_idx[i] = frame_num

    return {
        "csi": csi_frames,
        "kpts18": kpts18,
        "environment": environment,
        "sample": subject,
        "action": action,
        "frame_idx": frame_idx,
    }
```

- [ ] **Step 5: 更新 `_worker` 和 `main` 中的调用链**

更新 `_worker` 函数签名，透传 `gt_dir`：

```python
def _worker(args):
    trial_dir, gt_dir = args
    try:
        result = process_trial(Path(trial_dir), Path(gt_dir))
        label = f"{Path(trial_dir).parent.name}/{Path(trial_dir).name}"
        return label, result, None
    except Exception:
        label = f"{Path(trial_dir).parent.name}/{Path(trial_dir).name}"
        return label, None, traceback.format_exc()
```

在 `main()` 中更新调用：

```python
gt_dir = Path(args.gt)

# 单进程
result = process_trial(trial, gt_dir)

# 多进程
tasks = [(str(t), str(gt_dir)) for t in trials]
```

- [ ] **Step 5b: Phase 2 拼接前对 `all_data` 排序，确保确定性顺序**

无论单进程还是多进程，拼接前按 `(action, subject)` 排序，保证最终排列为 `action → subject → frame` 的层级顺序：

```python
print("Concatenating...")
# 确保多进程模式下顺序确定性
all_data.sort(key=lambda d: (d["action"], d["sample"]))
all_csi_raw  = np.concatenate([d["csi"] for d in all_data], axis=0).astype(np.float32)
all_kpts18   = np.concatenate([d["kpts18"] for d in all_data], axis=0).astype(np.float32)
```

- [ ] **Step 6: 移除 `derive_env`、`--train-subjects`，重构 Phase 2 归一化统计量**

删除 `derive_env` 函数（process_trial 中已内联 env 推导）。

删除 `--train-subjects` CLI 参数，Phase 2 归一化统计量改为基于**全部样本**计算，与训练时 `--env` 解耦：

```python
# DELETE these CLI arguments:
# parser.add_argument("--train-subjects", ...)

# DELETE: train_set = set(args.train_subjects)

# REPLACE the normalization statistics section:
print("Computing normalization statistics (all samples)...")
amp_min  = float(all_csi_raw.min())
amp_max  = float(all_csi_raw.max())
amp_mean = float(all_csi_raw.mean())
amp_std  = float(all_csi_raw.std())
print(f"  min={amp_min:.4f}  max={amp_max:.4f}  mean={amp_mean:.4f}  std={amp_std:.4f}")

# DELETE: train_mask, train_csi, n_train

# stats.json update — remove train_frames, keep total_frames:
stats = {
    "amplitude_min":  amp_min,
    "amplitude_max":  amp_max,
    "amplitude_mean": amp_mean,
    "amplitude_std":  amp_std,
    "total_frames":   n_total,
}
```

注意同时删除拼接阶段中不再需要的变量：

```python
# DELETE:
# n_train = int(np.isin(all_subjects.astype(str), list(train_set)).sum())
# print(f"Total: {n_total}, train: {n_train} ({n_train/n_total*100:.1f}%)")
```

- [ ] **Step 7: 运行 pylint 检查语法**

```bash
conda activate WiFiPose && python -c "import ast; ast.parse(open('scripts/build_memmap.py').read()); print('Syntax OK')"
```

预期：`Syntax OK`

---

### Task 2: 改造 `MemmapDataset` — 新增环境筛选与划分模式

**Files:**
- Modify: `data/memmap_dataset.py` (全文)

- [ ] **Step 1: 新增导入和常量**

在文件顶部新增 `SPLIT_RATIOS` 常量：

```python
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from data.heatmap_gt import build_pcm_paf


SPLIT_RATIOS = (0.7, 0.2, 0.1)

CSI_FILES = {
    "global_minmax": "csi_gminmax.npy",
    "global_zscore": "csi_gzscore.npy",
    "zscore": "csi_zscore.npy",
}
```

- [ ] **Step 2: 更新 `__init__` 签名**

在 `__init__` 中新增 `envs` 和 `split_mode` 参数，移除不再使用的 `envs`（旧的 `envs` 参数已存在但含义不同）、`train_subjects`、`test_subjects` 参数：

```python
def __init__(
    self,
    data_dir: str | Path,
    split: str = "train",
    envs: Iterable[str] | None = None,
    split_mode: str = "subject",
    random_val_ratio: float = 0.2,
    seed: int = 42,
    time_packets: int = 64,
    subcarrier_mode: str = "keep",
    normalize: str = "global_minmax",
    heatmap_size: int = 36,
    heatmap_sigma: float = 1.5,
    paf_width: float = 1.0,
    pose_range: tuple[float, float] = (-0.8, 0.8),
    build_targets: bool = True,
) -> None:
    if split not in {"train", "val", "test", "all"}:
        raise ValueError(f"split must be train/val/test/all, got {split}")
    if split_mode not in {"subject", "frame"}:
        raise ValueError(f"split_mode must be subject/frame, got {split_mode}")
    self.split = split
    self.split_mode = split_mode
    self.normalize = normalize
    self.heatmap_size = heatmap_size
    self.heatmap_sigma = heatmap_sigma
    self.paf_width = paf_width
    self.pose_range = pose_range
    self.build_targets = build_targets

    data_dir = Path(data_dir)

    if normalize not in CSI_FILES:
        raise ValueError(f"Unknown normalize mode: {normalize}, expected one of {list(CSI_FILES)}")

    self._csi = np.load(str(data_dir / CSI_FILES[normalize]), mmap_mode="r")
    self._kpts18 = np.load(str(data_dir / "ground_truth.npy"))

    meta = np.load(str(data_dir / "meta.npz"), allow_pickle=True)
    self._envs = meta["environment"]
    self._samples = meta["sample"]
    self._actions = meta["action"]

    self.indices = self._build_split(split, envs, split_mode, seed)
```

- [ ] **Step 3: 重写 `_build_split` 方法**

```python
def _build_split(
    self,
    split: str,
    envs: Iterable[str] | None,
    split_mode: str,
    seed: int,
) -> np.ndarray:
    env_set = set(envs) if envs else None
    num_total = len(self._samples)

    candidate_frame_indices: list[int] = []
    for i in range(num_total):
        if env_set is not None and str(self._envs[i]) not in env_set:
            continue
        candidate_frame_indices.append(i)

    if split == "all":
        return np.asarray(sorted(candidate_frame_indices), dtype=np.int64)

    if split_mode == "subject":
        return self._build_subject_split(candidate_frame_indices, split)
    else:
        return self._build_frame_split(candidate_frame_indices, split, seed)

def _build_subject_split(
    self,
    candidate_indices: list[int],
    split: str,
) -> np.ndarray:
    train_subjects = {f"S{i:02d}" for i in range(1, 8)}   # S01–S07
    val_subjects = {f"S{i:02d}" for i in range(8, 10)}     # S08–S09
    test_subjects = {"S10"}

    result: list[int] = []
    for idx in candidate_indices:
        subject = str(self._samples[idx])
        if split == "train" and subject in train_subjects:
            result.append(idx)
        elif split == "val" and subject in val_subjects:
            result.append(idx)
        elif split == "test" and subject in test_subjects:
            result.append(idx)
    return np.asarray(result, dtype=np.int64)

def _build_frame_split(
    self,
    candidate_indices: list[int],
    split: str,
    seed: int,
) -> np.ndarray:
    rng = random.Random(seed)
    shuffled = candidate_indices[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(round(total * SPLIT_RATIOS[0]))
    val_end = train_end + int(round(total * SPLIT_RATIOS[1]))

    if split == "train":
        selected = sorted(shuffled[:train_end])
    elif split == "val":
        selected = sorted(shuffled[train_end:val_end])
    else:
        selected = sorted(shuffled[val_end:])
    return np.asarray(selected, dtype=np.int64)
```

- [ ] **Step 4: 运行语法检查**

```bash
conda activate WiFiPose && python -c "import ast; ast.parse(open('data/memmap_dataset.py').read()); print('Syntax OK')"
```

---

### Task 3: 更新 `dataloader.py` — 工厂函数透传新参数

**Files:**
- Modify: `dataloader.py` (第36-72行)

- [ ] **Step 1: 更新 `create_memmap_data_loader` 签名**

```python
def create_memmap_data_loader(
    data_dir: str | Path,
    split: str,
    batch_size: int,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    seed: int = 42,
    envs: Iterable[str] | None = None,
    split_mode: str = "subject",
) -> DataLoader:
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split}")

    dataset = MemmapDataset(
        data_dir=data_dir,
        split=split,
        envs=envs,
        split_mode=split_mode,
        seed=seed,
        build_targets=False,
    )
    should_shuffle = shuffle if shuffle is not None else split == "train"
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

- [ ] **Step 2: 更新 `create_memmap_data_loaders` 签名**

```python
def create_memmap_data_loaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    envs: Iterable[str] | None = None,
    split_mode: str = "subject",
) -> dict[str, DataLoader]:
    return {
        split: create_memmap_data_loader(
            data_dir=data_dir,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            envs=envs,
            split_mode=split_mode,
        )
        for split in SPLIT_NAMES
    }
```

- [ ] **Step 3: 运行语法检查**

```bash
conda activate WiFiPose && python -c "import ast; ast.parse(open('dataloader.py').read()); print('Syntax OK')"
```

---

### Task 4: 更新 `train.py` — TrainConfig 与 CLI 新增参数

**Files:**
- Modify: `train.py` (第25-48行 TrainConfig，第260-399行 run_training/parse_args)

- [ ] **Step 1: 更新 `TrainConfig` dataclass**

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
    envs: list[str] | None = None
    split_mode: str = "subject"
```

- [ ] **Step 2: 更新 `run_training` 中 dataloader 创建调用**

在 `run_training` 函数中，将：

```python
loaders = create_memmap_data_loaders(
    data_dir=config.dataset_root,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    seed=config.seed,
)
```

改为：

```python
loaders = create_memmap_data_loaders(
    data_dir=config.dataset_root,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    seed=config.seed,
    envs=config.envs,
    split_mode=config.split_mode,
)
```

- [ ] **Step 3: 更新 `parse_args` 新增 CLI 参数**

在 `parse_args` 函数中新增 `--env` 和 `--split-mode` 参数：

```python
parser.add_argument("--env", type=str, nargs="+", default=None,
                    help="Environments to include (e.g. --env env1 env2). Default: all.")
parser.add_argument("--split-mode", type=str, default="subject", choices=["subject", "frame"],
                    help="Split mode: subject (7:2:1 per sample) or frame (7:2:1 global).")
```

- [ ] **Step 4: 更新 Config 构造，将 CLI 参数传入**

在 `main()` 中，将 `args.env` 和 `args.split_mode` 传入 `TrainConfig`：

```python
config = TrainConfig(
    dataset_root=args.dataset_root,
    output_dir=args.output_dir,
    axial_mode=args.axial_mode,
    decoder_type=args.decoder_type,
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    max_lr=args.max_lr,
    weight_decay=args.weight_decay,
    grad_clip_norm=args.grad_clip_norm,
    bone_loss_weight=args.bone_loss_weight,
    heatmap_size=args.heatmap_size,
    heatmap_sigma=args.heatmap_sigma,
    paf_width=args.paf_width,
    paf_loss_weight=args.paf_loss_weight,
    num_workers=args.num_workers,
    device=args.device,
    seed=args.seed,
    subset_size=args.subset_size,
    envs=args.env,
    split_mode=args.split_mode,
)
```

- [ ] **Step 5: 运行完整导入检查**

```bash
conda activate WiFiPose && python -c "from train import TrainConfig; print('Import OK')"
```

---

### Task 5: 编写数据集划分逻辑测试

**Files:**
- Create: `tests/test_memmap_dataset.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: 创建 `tests/__init__.py`**

```python
# tests package
```

- [ ] **Step 2: 编写测试文件 `tests/test_memmap_dataset.py`**

```python
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from data.memmap_dataset import MemmapDataset, SPLIT_RATIOS


def build_fake_memmap_dir(num_subjects: int = 40, num_actions: int = 27, frames_per: int = 5) -> Path:
    tmp = tempfile.mkdtemp(prefix="test_memmap_")
    root = Path(tmp)

    total = num_subjects * num_actions * frames_per
    csi = np.random.randn(total, 64, 3, 114).astype(np.float32)
    kpts18 = np.random.randn(total, 18, 2).astype(np.float32)

    environments = np.empty(total, dtype=object)
    samples = np.empty(total, dtype=object)
    actions = np.empty(total, dtype=object)

    idx = 0
    for subj_idx in range(1, num_subjects + 1):
        subj = f"S{subj_idx:02d}"
        env = f"env{(subj_idx - 1) // 10 + 1}"
        for act_idx in range(1, num_actions + 1):
            act = f"A{act_idx:02d}"
            for f in range(frames_per):
                environments[idx] = env
                samples[idx] = subj
                actions[idx] = act
                idx += 1

    np.save(str(root / "csi_gminmax.npy"), csi)
    np.save(str(root / "ground_truth.npy"), kpts18)
    np.savez(str(root / "meta.npz"), environment=environments, sample=samples, action=actions, frame_idx=np.arange(total))

    return root


class TestMemmapDatasetSubjectSplit:

    def test_subject_split_train_contains_s01_to_s07(self):
        root = build_fake_memmap_dir(frames_per=1)
        ds = MemmapDataset(root, split="train", envs=None, split_mode="subject", build_targets=False)
        samples_in_train = {str(ds._samples[i]) for i in ds.indices}
        for s in ["S01", "S02", "S03", "S04", "S05", "S06", "S07"]:
            assert s in samples_in_train

    def test_subject_split_val_contains_s08_s09(self):
        root = build_fake_memmap_dir(frames_per=1)
        ds = MemmapDataset(root, split="val", envs=None, split_mode="subject", build_targets=False)
        samples_in_val = {str(ds._samples[i]) for i in ds.indices}
        assert "S08" in samples_in_val
        assert "S09" in samples_in_val

    def test_subject_split_test_contains_s10_only(self):
        root = build_fake_memmap_dir(frames_per=1)
        ds = MemmapDataset(root, split="test", envs=None, split_mode="subject", build_targets=False)
        samples_in_test = {str(ds._samples[i]) for i in ds.indices}
        assert samples_in_test == {"S10"}

    def test_subject_split_no_overlap(self):
        root = build_fake_memmap_dir(frames_per=3)
        train_ds = MemmapDataset(root, split="train", envs=None, split_mode="subject", build_targets=False)
        val_ds = MemmapDataset(root, split="val", envs=None, split_mode="subject", build_targets=False)
        test_ds = MemmapDataset(root, split="test", envs=None, split_mode="subject", build_targets=False)
        train_set = set(train_ds.indices)
        val_set = set(val_ds.indices)
        test_set = set(test_ds.indices)
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_subject_split_env_filter(self):
        root = build_fake_memmap_dir(frames_per=1)
        ds = MemmapDataset(root, split="train", envs=["env1"], split_mode="subject", build_targets=False)
        envs_in_ds = {str(ds._envs[i]) for i in ds.indices}
        assert envs_in_ds == {"env1"}


class TestMemmapDatasetFrameSplit:

    def test_frame_split_ratios_approximate(self):
        root = build_fake_memmap_dir(frames_per=5)
        ds_all = MemmapDataset(root, split="all", envs=None, split_mode="frame", build_targets=False)
        total = len(ds_all)
        ds_train = MemmapDataset(root, split="train", envs=None, split_mode="frame", build_targets=False)
        ds_val = MemmapDataset(root, split="val", envs=None, split_mode="frame", build_targets=False)
        ds_test = MemmapDataset(root, split="test", envs=None, split_mode="frame", build_targets=False)
        assert len(ds_train) + len(ds_val) + len(ds_test) == total
        assert abs(len(ds_train) / total - SPLIT_RATIOS[0]) < 0.05
        assert abs(len(ds_val) / total - SPLIT_RATIOS[1]) < 0.05

    def test_frame_split_no_overlap(self):
        root = build_fake_memmap_dir(frames_per=5)
        train_ds = MemmapDataset(root, split="train", envs=None, split_mode="frame", build_targets=False)
        val_ds = MemmapDataset(root, split="val", envs=None, split_mode="frame", build_targets=False)
        test_ds = MemmapDataset(root, split="test", envs=None, split_mode="frame", build_targets=False)
        train_set = set(train_ds.indices)
        val_set = set(val_ds.indices)
        test_set = set(test_ds.indices)
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_frame_split_reproducible(self):
        root = build_fake_memmap_dir(frames_per=5)
        ds1 = MemmapDataset(root, split="train", envs=None, split_mode="frame", seed=42, build_targets=False)
        ds2 = MemmapDataset(root, split="train", envs=None, split_mode="frame", seed=42, build_targets=False)
        assert np.array_equal(ds1.indices, ds2.indices)

    def test_frame_split_env_filter(self):
        root = build_fake_memmap_dir(frames_per=1)
        ds = MemmapDataset(root, split="train", envs=["env2"], split_mode="frame", build_targets=False)
        envs_in_ds = {str(ds._envs[i]) for i in ds.indices}
        assert envs_in_ds == {"env2"}


class TestMemmapDatasetItem:

    def test_getitem_returns_csi_and_keypoints(self):
        root = build_fake_memmap_dir(frames_per=1)
        ds = MemmapDataset(root, split="train", envs=None, split_mode="subject", build_targets=False)
        item = ds[0]
        assert torch.is_tensor(item["csi"])
        assert torch.is_tensor(item["kpts18"])
        assert item["csi"].shape == (64, 3, 114)
        assert item["kpts18"].shape == (18, 2)
        assert "meta" in item
```

- [ ] **Step 3: 运行测试**

```bash
conda activate WiFiPose && pytest tests/test_memmap_dataset.py -v
```

预期：所有测试通过。

---

### Task 6: 更新 AGENTS.md

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: 更新命令示例**

将文档中的命令更新为新的接口：

```markdown
Build an NPY memmap dataset:

```powershell
python scripts\build_memmap.py --src D:\path\to\raw\dataset --gt D:\path\to\ground_truth_npy --output-dir data\mmfi_pose --workers 8
```

Run a quick training sanity check:

```powershell
python train.py --dataset-root data\mmfi_pose --epochs 5 --subset-size 32 --output-dir outputs\sanity --env env1 --split-mode subject
```

Run the default training configuration:

```powershell
python train.py --dataset-root data\mmfi_pose --epochs 50 --batch-size 64 --output-dir outputs\train --env env1 --split-mode subject
```

Frame-level split training:

```powershell
python train.py --dataset-root data\mmfi_pose --epochs 50 --batch-size 64 --output-dir outputs\train_frame --env env1 --split-mode frame
```
```

---

### Task 7: 端到端验证

- [ ] **Step 1: 运行完整测试套件**

```bash
conda activate WiFiPose && pytest tests/test_memmap_dataset.py -v
```

- [ ] **Step 2: 运行语法导入检查**

```bash
conda activate WiFiPose && python -c "
from data.memmap_dataset import MemmapDataset
from dataloader import create_memmap_data_loader, create_memmap_data_loaders
from train import TrainConfig
print('All imports OK')
"
```

- [ ] **Step 3: 检查 `build_memmap.py` 导入**

```bash
conda activate WiFiPose && python -c "import ast; ast.parse(open('scripts/build_memmap.py').read()); print('Syntax OK')"
```

---

### 自审清单

**1. Spec 覆盖检查：**
- [x] 数据路径与结构修正：Task 1 迁移 ground truth 数据源
- [x] 环境元数据管理系统：Task 1 中 env 作为元数据存储，Task 2 中 envs 参数筛选
- [x] 数据加载与预处理流程优化：Task 1 重构 process_trial，Task 2 重构 _build_split
- [x] 训练配置与接口更新：Task 3/4 新增参数透传
- [x] 测试与验证计划：Task 5 测试划分逻辑，Task 7 端到端验证

**2. Placeholder 扫描：** 无 TBD/TODO/fill-in

**3. 类型一致性：** `envs` 统一为 `Iterable[str] | None`，`split_mode` 统一为 `str`，默认值 `"subject"`