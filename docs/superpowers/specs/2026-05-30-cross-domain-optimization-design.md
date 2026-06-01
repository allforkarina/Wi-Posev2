# Cross-Domain Few-Shot Performance Optimization

**Date**: 2026-05-30
**Status**: Draft

## Problem

当前 Tier 2 finetune（env1→env2）卡在 PCK@0.2=80.7%，MPJPE=0.153。主要瓶颈：

- **脚踝关节**：right ankle PCK=40.0%, left ankle PCK=40.4%
- **困难动作**：A15 PCK=58.6%, A19 PCK=62.5%, A16 PCK=66.0%
- **高方差**：per-joint var_ratio 0.46~0.81，腕部(wrist)最高
- 不是 mean-pose 问题（var_ratio 都在健康范围）

## Approach 1: Frequency-Domain CSI Augmentation ⭐⭐⭐

**改动量**: 新增 ~30 行，`dataloader.py` 中添加 `CSI augment` transform

**原理**: 跨域差异来自不同房间的多径反射和幅度衰减。在 few-shot 训练时给 target 域样本加扰动，模拟更多"类 target 域"变体，提升模型对域变化的鲁棒性。

### Augmentation Pipeline（仅 finetune 阶段生效）

| 增强 | 操作 | 概率 | 参数范围 |
|------|------|------|---------|
| 幅度抖动 | `csi * U[a, b]` per-sample | 1.0 | `[0.7, 1.3]` |
| 天线 mask | 随机置零 1 根天线 | 0.3 | 3→2 天线 |
| 时间 mask | 随机 mask 连续帧段 | 0.3 | 4–8 帧 |
| 子载波 mask | 随机 mask 子载波段 | 0.3 | 8–16 子载波 |

### Implementation

- 在 `dataloader.py` 新增 `CSI augment(csi: Tensor) -> Tensor` 函数
- 在 `memmap_collate_fn` 中调用（仅当 `training=True` 时）
- CLI 新增 `--csi-augment` flag（默认 off，可单独开关消融）

---

## Approach 2: Hard-Sample Weighted Loss ⭐⭐

**改动量**: 新增 ~15 行，`train.py` 中修改 `compute_losses()`

**原理**: 脚踝(40%)和 A15(58%)贡献了巨大的 loss 但被大批量平均抹平。给高误差样本加权让模型更关注这些困难 case。

### 权重方案

```python
# Per-sample MPJPE 作为困难度指标
per_sample_mpjpe = torch.linalg.vector_norm(prediction - target, dim=-1).mean(dim=-1)  # [B]
# 权重 = exp(mpjpe / temperature) 归一化
weight = torch.exp(per_sample_mpjpe / temperature)  # temperature ≈ 0.1
weight = weight / weight.mean()  # 保持整体 loss scale 不变
loss = (loss_per_sample * weight).mean()
```

### Implementation

- 在 `compute_losses()` 中添加 `hard_sample_weight: bool = False` 参数
- `TrainConfig` 新增 `hard_sample_weight: bool = False`
- CLI 新增 `--hard-sample-weight` flag

---

## Approach 3: Multi-Source Domain Training ⭐⭐⭐

**改动量**: 0 行代码，仅 CLI 参数变更

**原理**: 当前只用了 env1 作为源域。MM-Fi 数据集有多个环境，用多个 env 作为 source 可以扩展模型见过的域分布，提升泛化能力。

### 用法

```powershell
# 当前：单源
--source-envs env1

# 改为：多源（例如用 env1+env3+env5）
--source-envs env1 env3 env5
```

### 注意事项

- 需要确认数据集中有哪些 env
- 多源训练时间更长（源域数据量翻倍）
- 需要重新训练 source_only baseline（旧 checkpoint 不兼容）

---

## Approach 4: Domain-Adversarial Training (DANN) ⭐⭐⭐⭐

**改动量**: 新增 ~80 行，`train.py` + 新增 `models/wiflow_domain_classifier.py`

**原理**: 在 encoder 输出层加一个域分类器（判断样本来自源域还是目标域），通过梯度反转层(Gradient Reversal Layer)迫使 encoder 学习域不变特征。

### Architecture

```
CSI Input → SpatialEncoder → AxialEncoder → [GradReversal] → DomainClassifier → src/tgt
                                                ↓
                                            Decoder → Keypoints
```

- **DomainClassifier**: 2-layer MLP: `[256*29*16] → 128 → 2`，输出源/目标二分类
- **GradReversal Layer**: forward 时恒等，backward 时梯度取反 × λ
- **λ 衰减**: `λ(epoch) = λ_max * (2/(1+exp(-10*epoch/max_epochs)) - 1)` （从 0 逐渐增大到 λ_max）

### Loss

```
总 loss = coord_loss + bone_loss + λ * domain_classification_loss
```

域分类 loss 只对源域样本计算（因为源域样本有标准监督，微调时目标域少量样本无监督）。

### Implementation

- 新增 `models/wiflow_domain_classifier.py`：`GradientReversalLayer` + `DomainClassifier`
- `WiFlowModel` 新增 `domain_classifier` 子模块（可插拔）
- `train.py` finetune 循环计算域分类 loss
- CLI 新增 `--domain-adversarial` + `--domain-lambda` 参数

---

## Recommended Execution Order

```
1. Multi-Source → 2. CSI Augment → 3. Hard-Sample Weight → 4. Domain-Adversarial
```

- **方案 3 先做**：零代码改动，快速验证多源是否有效。如果多源能提 5%+，后续增强都建在更强的 baseline 上
- **方案 1 次之**：改动小，针对域差根因
- **方案 2 作为补充**：与方案 1 叠加解决困难样本
- **方案 4 兜底**：改动最大，但直接针对域不变特征这一核心问题

---

## Expected Effects

| 方案 | 预期 PCK@0.2 提升 | 风险 |
|------|-------------------|------|
| 多源域 | +3~8% | 训练时间翻倍 |
| CSI Augment | +2~5% | 增强过强可能引入噪声 |
| Hard-Sample Weight | +1~3% | 可能过拟合困难样本 |
| Domain-Adversarial | +3~10% | 训练不稳定，需调参 |
