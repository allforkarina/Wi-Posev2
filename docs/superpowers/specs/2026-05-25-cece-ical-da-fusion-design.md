# WiFlow + AdaPose DA 融合设计文档

> 日期: 2026-05-25 | 状态: 设计确认

## 概述

将 AdaPose 的两个跨域泛化机制（CECE + ICAL）以最小侵入方式融合到 WiFlow 架构中，形成支持跨环境域适应的姿态估计系统。

### 设计原则

- **模块解耦**：CECE/ICAL 作为独立模块，训练循环层编排 DA 逻辑
- **模型不变**：`WiFlowModel.forward()` 接口零改动，推理/评估路径不受影响
- **纯 DA 升级**：不保留单域兼容模式，当前仓库专做 cross-env DA

### 核心决策

| 决策点 | 选择 |
|--------|------|
| 域划分方式 | 按环境（environment）划分 |
| 目标域标注 | 有标注（supervised DA） |
| 向后兼容 | 不需要，纯 DA 升级 |
| Decoder 支持 | joint + hierarchical（参数选择） |
| ICAL 特征层 | Axial Encoder 输出，GAP 池化 |
| CECE 作用对象 | 源域和目标域同时重加权 |

---

## 模块设计

### 1. CECEModule — 跨环境通道增强

**文件**: `models/cece.py`

**职责**: 利用源域/目标域特征图的通道级余弦相似度，对两域特征同时做通道重加权，抑制环境噪声通道。

**接口**:

```python
class CECEModule(nn.Module):
    def __init__(self, num_channels: int = 256) -> None
    def forward(self, src_feat: Tensor[B,C,H,W], tgt_feat: Tensor[B,C,H,W]) -> tuple[Tensor, Tensor]
```

**算法**:
1. batch 内平均 → `[C, H, W]` 域级代表特征图
2. 空间展开 → `[C, H*W]`，逐通道计算余弦相似度 → `[C]`
3. 线性映射 `(cos_sim + 1) / 2` → `[C]`，值域 [0, 1]
4. 广播回 `[1, C, 1, 1]`，同时乘源域和目标域特征

**特性**: 无参数、无状态、测试时不调用

### 2. ICAL — 实例级一致性对齐损失

**文件**: `train.py::compute_ical_loss`

**职责**: 用姿态相似度矩阵对特征距离加权，让"姿态相似的源-目标对"获得更强对齐。

**接口**:

```python
def compute_ical_loss(
    f_s:       Tensor[B, D],   # 源域 CECE 重加权后特征 (Axial Encoder GAP)
    f_t:       Tensor[B, D],   # 目标域 CECE 重加权后特征
    y_s_gt:    Tensor[B,18,2], # 源域真值（非预测值，避免冷启动）
    y_t_pred:  Tensor[B,18,2], # 目标域预测值
    sigma_pose: float = 0.5,
) -> Tensor  # scalar
```

**算法**:
1. 姿态距离矩阵 `D[i,j] = ||y_s_gt[i] - y_t_pred[j]||` → `[B, B]`
2. 权重 `W[i,j] = exp(-D[i,j] / sigma_pose)`，行归一化：`W / sum(W, dim=1)`
3. 特征距离 `F[i,j] = ||f_s[i] - f_t[j]||²`
4. `ICAL = mean(W * F)`（除以 B 稳定量级）

### 3. 训练循环 run_da_epoch

**文件**: `train.py`

**职责**: 编排双域前向 + CECE + ICAL + 双域监督损失。

**流程**:
```
for batch_t in target_loader:
    batch_s = next(source_iter)   # 源域 cycle，每 epoch 重建迭代器

    # 前向：手动拆解 WiFlowModel 三步
    feat_s = model.axial_encoder(model.spatial_encoder(x_s))
    feat_t = model.axial_encoder(model.spatial_encoder(x_t))
    feat_s_ce, feat_t_ce = cece(feat_s, feat_t)
    y_s = model.decoder(feat_s_ce)
    y_t = model.decoder(feat_t_ce)

    # 损失
    loss_sup_s = compute_losses(y_s, kp_s_gt)
    loss_sup_t = compute_losses(y_t, kp_t_gt)
    loss_ical = compute_ical_loss(feat_s_ce.mean([2,3]), feat_t_ce.mean([2,3]), kp_s_gt, y_t)
    loss = (loss_sup_s + loss_sup_t) / 2 + actual_alpha * loss_ical

    # 优化
    loss.backward() → clip_grad → optimizer.step() → scheduler.step()
```

**ICAL warmup**: `actual_alpha = alpha * min(1.0, epoch / warmup_epochs)`

**源域迭代器管理**: 在 epoch 外层创建 `source_iter = iter(source_loader)`，内部 `except StopIteration` 时重建，保证跨 epoch 重新 shuffle 避免固定配对。

### 4. 验证循环 run_val_epoch

**文件**: `train.py`

**职责**: 仅迭代目标域 validation loader，计算监督损失和指标。

**与 run_da_epoch 分离**: 验证时不需要源域数据，也不需要 ICAL（虽可计算用于诊断，但不用作 early stopping 判据）。best checkpoint 按 `val_target_mpjpe` 选择。

### 5. 数据管道

**文件**: `dataloader.py::create_da_data_loaders`

**接口**:

```python
def create_da_data_loaders(
    data_dir: str | Path,
    source_envs: Sequence[str],
    target_envs: Sequence[str],
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> dict:
```

**返回**:
- `"source_train"`: 源域全部数据（`split="all"`，不拆 val），shuffle=True
- `"target_train"`: 目标域训练集（`random_val_ratio=0.2`），shuffle=True
- `"target_val"`: 目标域验证集
- `"target_test"`: 目标域测试集

### 6. 配置层 TrainConfig

**新增字段**:

```python
@dataclass(frozen=True)
class TrainConfig:
    # 保留字段
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

    # 新增 DA 字段
    source_envs: tuple[str, ...] = ("lab",)
    target_envs: tuple[str, ...] = ("corridor",)
    alpha: float = 0.1
    ical_warmup_epochs: int = 5
    cece_enabled: bool = True
```

**CSV 日志新增列**: `train_source_loss`, `train_target_loss`, `train_ical`, `train_source_mpjpe`, `train_target_mpjpe`, `alpha`

### 7. CLI 参数扩展

新增参数:
- `--source-envs`, nargs="+", default=["lab"]
- `--target-envs`, nargs="+", default=["corridor"]
- `--alpha`, type=float, default=0.1
- `--ical-warmup-epochs`, type=int, default=5
- `--no-cece`, action="store_true"（禁用 CECE）

---

## 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `models/cece.py` | CECEModule 实现 |
| 修改 | `models/__init__.py` | 导出 CECEModule |
| 修改 | `train.py` | TrainConfig 扩展、run_da_epoch、run_val_epoch、compute_ical_loss、run_training 重写、CLI 扩展 |
| 修改 | `dataloader.py` | 新增 create_da_data_loaders |
| 不变 | `models/wiflow_*.py`, `models/skeleton.py`, `pose_targets.py`, `data/*`, `eval.py`, `evaluation/*` | 零改动 |

---

## 训练命令示例

```powershell
# 基础 DA 训练
python train.py --dataset-root data\mmfi_pose --source-envs lab --target-envs corridor --epochs 50 --batch-size 64

# 禁用 CECE，纯 ICAL
python train.py --dataset-root data\mmfi_pose --source-envs lab --target-envs corridor --no-cece

# 调整 ICAL 强度
python train.py --dataset-root data\mmfi_pose --source-envs lab --target-envs corridor --alpha 0.5 --ical-warmup-epochs 10

# 使用 hierarchical decoder
python train.py --dataset-root data\mmfi_pose --source-envs lab --target-envs corridor --decoder-type hierarchical
```