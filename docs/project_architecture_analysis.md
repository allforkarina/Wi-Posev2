# WiFlow 项目全面架构分析

## 一、项目总体架构

### 1.1 项目概览

WiFlow 是一个基于 **WiFi CSI（通道状态信息）单帧人体姿态估计** 的深度学习项目。输入为 3 天线 × 114 子载波 × 64 时间步的 CSI 振幅数据，输出为 OpenPose 18 关键点的 2D 坐标。

### 1.2 目录与模块组织

```
WiFlow/
├── train.py                          # 训练入口
├── eval.py                           # 评估/可视化入口
├── dataloader.py                     # DataLoader 工厂
├── pose_targets.py                   # PCM/PAF 在线目标合成 (Torch)
├── models/                           # 模型核心包
│   ├── __init__.py                   # 统一导出
│   ├── skeleton.py                   # 骨架拓扑定义 (19 骨骼边)
│   ├── wiflow_model.py               # 顶层模型组装
│   ├── wiflow_spatial_encoder.py     # CNN 空间编码器
│   ├── wiflow_axial_encoder.py       # 轴向注意力编码器
│   ├── wiflow_joint_decoder.py       # 关节交叉注意力解码器
│   ├── wiflow_hierarchical_joint_decoder.py  # 层级关节解码器
│   └── wiflow_heatmap_decoder.py     # MSFN 热图解码器 + PAPM
├── data/                             # 数据层
│   ├── memmap_dataset.py             # NPY memmap 数据集
│   └── heatmap_gt.py                 # 离线 PCM/PAF 生成 (NumPy)
├── scripts/
│   ├── build_memmap.py               # 数据集构建脚本
│   └── diagnose_loss.py              # 损失诊断工具
└── tests/                            # 测试 (11 有效 + 4 遗留)
```

### 1.3 系统架构图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           WiFlow 系统架构                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────┐     ┌──────────────────────────┐               │
│  │   scripts/           │     │   CLI Entry Points        │               │
│  │   build_memmap.py ───┼────►│   train.py  eval.py       │               │
│  │   (MAT→NPY 转换)     │     │   dataloader.py           │               │
│  └──────────────────────┘     └──────────┬───────────────┘               │
│                                         │                                │
│                    ┌────────────────────┼────────────────────┐           │
│                    │                    │                    │           │
│                    ▼                    ▼                    ▼           │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────┐   │
│  │  data/ 层             │  │  models/ 层           │  │ 辅助模块      │   │
│  │  MemmapDataset        │  │                      │  │              │   │
│  │  heatmap_gt.py        │  │  WiFlowModel          │  │ pose_targets │   │
│  │  (零拷贝数据访问)       │  │   ├─ SpatialEncoder   │  │ (在线目标合成)│   │
│  └──────────┬────────────┘  │   ├─ AxialEncoder     │  │              │   │
│             │                │   ├─ JointDecoder     │  │ train.py 中  │   │
│             │                │   ├─ HierarchicalDec  │  │ compute_loss │   │
│             │                │   └─ MSFNDecoder      │  │ compute_metr │   │
│             │                └──────────────────────┘  └──────────────┘   │
│             │                                                            │
│  ┌──────────▼────────────────────────────────────────────────────────┐   │
│  │  NPY Memmap 文件系统                                               │   │
│  │  csi_gminmax.npy / csi_gzscore.npy / csi_zscore.npy                │   │
│  │  ground_truth.npy / meta.npz / stats.json                          │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  数据流:                                                                  │
│  NPY → MemmapDataset → collate_fn (permute) → DataLoader                  │
│  → WiFlowModel (Spatial→Axial→Decoder) → keypoints/pcm+paf               │
│  → Loss/Metrics → Optimizer(AdamW) + Scheduler(OneCycleLR)               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 二、模型内部架构（WiFlowModel）

```
CSI Input: [B, 3, 114, 64]
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  WiFlowSpatialEncoder (CNN 下采样)                     │
│                                                      │
│  antenna_mixer: Conv2d(3→3, 1×1)  ← 天线间信息融合    │
│  feature_stem:  Conv2d(3→32, 3×5)  ← 初始特征提取     │
│  block1: ResDown(32→64,   stride=2) → [B,64,57,32]  │
│  block2: ResDown(64→128,  stride=2) → [B,128,29,16] │
│  block3: ResDown(128→128, stride=1) → [B,128,29,16] │
│                                                      │
│  ※ 对称下采样: 时间轴 64→32→16, 子载波轴 114→57→29     │
└──────────────────────┬───────────────────────────────┘
                       │ [B, 128, 29, 16]
                       ▼
┌──────────────────────────────────────────────────────┐
│  WiFlowAxialEncoder (轴向自注意力)                      │
│                                                      │
│  mode: spatial_then_temporal / temporal_then_spatial │
│        parallel_sum / parallel_concat                 │
│                                                      │
│  spatial_attention:  29 tokens (子载波维度)             │
│       + residual + LayerNorm                         │
│  temporal_attention: 16 tokens (时间维度)              │
│       + residual + LayerNorm                         │
│  channel_projection: 1×1 conv, 128→256               │
│                                                      │
│  ※ 避免 29×16=464 token 的全注意力 O(N²) 代价           │
└──────────────────────┬───────────────────────────────┘
                       │ [B, 256, 29, 16]
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  解码器 (三选一)                                                   │
│                                                                  │
│  ┌─────────────────────┐  ┌────────────────────┐  ┌───────────┐ │
│  │ joint (默认)         │  │ hierarchical       │  │ heatmap   │ │
│  │ WiFlowJointDecoder  │  │ HierarchicalDec    │  │ _msfn     │ │
│  │                     │  │                    │  │           │ │
│  │ 18 learnable queries │  │ 三阶段渐进式:       │  │ MSFN 多阶段│ │
│  │ ×3 cross-attn layers│  │ Stage0: 躯干+头部   │  │ 热图回归   │ │
│  │ + GNN 骨架约束       │  │ Stage1: 四肢        │  │ + PAPM    │ │
│  │ + Self-Attention    │  │ Stage2: 面部        │  │ 特征调制   │ │
│  │ + MLP→[B,18,2]      │  │ + GNN + SA + MLP   │  │           │ │
│  └─────────────────────┘  └────────────────────┘  └───────────┘ │
│                                                                  │
│  输出: [B, 18, 2]             输出: [B, 18, 2]       输出: dict  │
│                                                        {         │
│                                                          keypoints,
│                                                          stages
│                                                        }         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 三、train.py 深入剖析

### 3.1 训练配置 (TrainConfig)

| 类别 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| 路径 | `dataset_root` | — | NPY memmap 数据集目录 |
| 路径 | `output_dir` | `"outputs/train"` | 日志和检查点输出目录 |
| 模型 | `axial_mode` | `"spatial_then_temporal"` | 轴向编码器模式 |
| 模型 | `decoder_type` | `"joint"` | 解码器类型 |
| 模型 | `heatmap_size` | `36` | 热图分辨率 |
| 训练 | `epochs` | `50` | 训练轮数 |
| 训练 | `batch_size` | `64` | 批次大小 |
| 训练 | `lr` | `2e-5` | 初始学习率 |
| 训练 | `max_lr` | `5e-4` | OneCycleLR 峰值学习率 |
| 正则 | `weight_decay` | `5e-4` | AdamW 权重衰减 |
| 正则 | `grad_clip_norm` | `1.0` | 梯度裁剪阈值 |
| 损失 | `bone_loss_weight` | `0.5` | 骨骼长度损失权重 |
| 损失 | `heatmap_sigma` | `1.5` | PCM 高斯sigma |
| 损失 | `paf_width` | `1.0` | PAF 向量场宽度 |
| 损失 | `paf_loss_weight` | `1.0` | PAF 损失权重 |
| 系统 | `num_workers` | `4` | DataLoader 工作进程数 |
| 系统 | `device` | `"cuda"` | 计算设备 |
| 系统 | `seed` | `42` | 随机种子 |
| 系统 | `subset_size` | `None` | 数据集子集大小(调试用) |

### 3.2 模型初始化

```python
model = WiFlowModel(
    input_channels=3,
    axial_mode=config.axial_mode,       # "spatial_then_temporal"
    decoder_type=config.decoder_type,    # "joint" / "hierarchical" / "heatmap_msfn"
    heatmap_size=config.heatmap_size,    # 36
).to(device)
```

模型组装后立即执行 **sanity check**：取第一个 batch 做前向推理，验证输出形状与标签形状一致。若不一致则抛出 `ValueError`，防止静默的形状不匹配。

### 3.3 损失函数设计（核心机制）

损失函数根据解码器类型采用**双轨分支策略**，通过 `isinstance(prediction, Mapping)` 判断：

```
compute_losses(prediction, target, ...)
         │
         │ isinstance(prediction, Mapping)?
         │
    ┌────┴────┐
    │ YES     │ NO
    │ (heatmap│ (joint/hierarchical)
    │  _msfn) │
    ▼         ▼
┌──────────────────────┐  ┌──────────────────────────┐
│ 热图监督模式            │  │ 坐标回归模式               │
│                      │  │                          │
│ build_pcm_paf_targets│  │ L1(pred, target)         │
│   → pcm_gt, paf_gt   │  │   = coord_loss           │
│                      │  │                          │
│ for stage in stages: │  │ bone_length_loss()       │
│   pcm_mse += MSE(    │  │   → 骨骼长度L1           │
│     stage.pcm,       │  │   = bone_loss            │
│     pcm_gt)          │  │                          │
│   paf_mse += MSE(    │  │ total = coord_loss       │
│     stage.paf,       │  │   + 0.5 * bone_loss      │
│     paf_gt)          │  │                          │
│                      │  │                          │
│ total = pcm_mse      │  │                          │
│   + paf_weight*      │  │                          │
│     paf_mse          │  │                          │
└──────────────────────┘  └──────────────────────────┘
```

**bone_length_loss 实现细节：**

对每条 OPENPOSE_BONE_EDGES 定义的骨骼边 `(i, j)`，计算预测骨骼长度与真值骨骼长度的 L1 损失：

```python
bone_length_loss(pred, target):
    pred_lengths  = ||pred[i] - pred[j]||   # 19 条边的预测长度
    target_lengths = ||target[i] - target[j]||  # 19 条边的真值长度
    return L1(pred_lengths, target_lengths)
```

这一损失项对骨架拓扑信息进行显式约束，迫使模型不仅学习关键点的绝对位置，还学习骨骼长度的结构一致性。

### 3.4 优化器与学习率调度

**优化器: AdamW**

| 参数 | 值 | 说明 |
|------|-----|------|
| `lr` | `2e-5` | 初始学习率 |
| `weight_decay` | `5e-4` | 解耦权重衰减（真正的 L2 正则化） |
| `β₁, β₂` | `0.9, 0.999` | PyTorch 默认值 |
| `ε` | `1e-8` | PyTorch 默认值 |

**学习率调度器: OneCycleLR（单周期余弦退火）**

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_lr` | `5e-4` | 峰值学习率（初始 lr 的 25 倍） |
| `pct_start` | `0.3` | 前 30% 步数线性 warmup |
| `anneal_strategy` | `"cos"` | 余弦退火 |
| `div_factor` | `max_lr / lr ≈ 25` | 初始lr与峰值lr比率 |
| `final_div_factor` | `1000` | 最终 lr = 初始 lr / 1000 ≈ `2e-8` |
| `steps_per_epoch` | `len(train_loader)` | 每个 batch 都执行 `scheduler.step()` |

**学习率曲线示意：**

```
lr
│
│  max_lr=5e-4 ┤        ╱╲
│              │       ╱  ╲
│              │      ╱    ╲
│              │     ╱      ╲_________ cos退火
│              │    ╱                 ╲
│  lr=2e-5 ────┤───╱ warmup            ╲
│              │  ╱  (30%)              ╲___
│              │ ╱                            ╲___ final_lr≈2e-8
│              └──────────────────────────────────────► steps
│              0      30%                   100%
│                    training steps
```

### 3.5 迭代训练流程 (run_epoch)

`run_epoch()` 函数同时处理训练和验证：

```python
def run_epoch(model, loader, config, device, optimizer=None, scheduler=None):
    is_training = optimizer is not None
    model.train(is_training)

    for batch in loader:
        model_input, target = prepare_model_input(batch, device)

        with torch.set_grad_enabled(is_training):
            prediction = model(model_input)
            losses = compute_losses(prediction, target, ...)
            keypoint_prediction = extract_prediction_keypoints(prediction)
            metrics = compute_metrics(keypoint_prediction.detach(), target)

        if is_training:
            optimizer.zero_grad(set_to_none=True)    # 置None比 fill_(0) 更高效
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(          # 梯度裁剪
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()                         # 每个 batch 更新 lr

        # 加权平均累积
        totals[name] += value * batch_size
    return totals / sample_count
```

训练与验证的切换通过 `optimizer is not None` 实现，验证时传入 `optimizer=None` 则跳过梯度计算和参数更新。

### 3.6 梯度计算与裁剪

梯度裁剪使用 `torch.nn.utils.clip_grad_norm_`，将全局梯度范数限制在 `max_norm=1.0` 以内，防止梯度爆炸。

梯度清零使用 `optimizer.zero_grad(set_to_none=True)` 而非 `zero_grad()`，将梯度张量直接置为 `None`，节省内存和后续的加法开销。

### 3.7 正则化措施汇总

| 措施 | 机制 | 配置 | 作用层级 |
|------|------|------|---------|
| **Weight Decay** | AdamW 解耦权重衰减 | `5e-4` | 所有可学习参数 |
| **Gradient Clipping** | `clip_grad_norm_` | `max_norm=1.0` | 梯度范数截断 |
| **Bone Loss** | 骨骼长度 L1 约束 | `weight=0.5` | 输出层结构化先验 |
| **OneCycleLR** | 大学习率 + 余弦退火 | `max_lr=5e-4` | 隐式泛化正则 |
| **PAF Loss** | 局部方向场监督 | `weight=1.0` | 仅 heatmap_msfn 模式 |

### 3.8 早停与模型选择策略

项目**没有实现传统的基于 patience 的早停机制**，而是采用"全程训练 + 事后择优"策略：

- 总是保存 `last.pth`（最新 checkpoint）
- `val_mpjpe` 改善时 → 保存 `best_val_mpjpe.pth`
- `val_pck_0_2` 改善时 → 保存 `best_val_pck_0_2.pth`

每个 checkpoint 包含完整恢复所需的所有状态：`model_state_dict`、`optimizer_state_dict`、`scheduler_state_dict`、`epoch`、`best_metric`、`train_config`。

### 3.9 评价指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **MPJPE** | `mean(‖predᵢ - targetᵢ‖)` | 平均每关节位置误差 |
| **PCK@τ** | `mean(‖predᵢ - targetᵢ‖ < scale × τ)` | 正确关键点百分比 |
| **Torso Scale** | `‖right_shoulder(2) - left_hip(11)‖` | 用于 PCK 归一化 |

PCK 支持 5 个阈值：`(0.1, 0.2, 0.3, 0.4, 0.5)`。

### 3.10 train.py 核心流程图

```
                        ┌──────────┐
                        │  main()  │
                        └────┬─────┘
                             │
                        ┌────▼─────┐
                        │ 解析 CLI  │
                        │ 参数      │
                        └────┬─────┘
                             │ TrainConfig(**vars(args))
                        ┌────▼─────┐
                        │ run_     │
                        │ training │
                        └────┬─────┘
                             │
                  ┌──────────┼──────────┐
                  ▼          ▼          ▼
           ┌──────────┐ ┌────────┐ ┌──────────┐
           │ 设置种子  │ │选设备   │ │创建输出   │
           │ seed=42  │ │cuda/cpu│ │目录       │
           └────┬─────┘ └───┬────┘ └────┬─────┘
                │            │           │
                └────────────┼───────────┘
                             │
                    ┌────────▼────────┐
                    │ create_memmap_  │
                    │ data_loaders()  │
                    │ → train/val/test│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌──────────────┐
     │maybe_subset│  │maybe_subset│  │ WiFlowModel()│
     │(train)     │  │(val)       │  │ → .to(device)│
     └─────┬──────┘  └─────┬──────┘  └──────┬───────┘
           │               │                 │
           │               │    ┌────────────▼───────────┐
           │               │    │ AdamW(model.parameters) │
           │               │    │ lr=2e-5, wd=5e-4       │
           │               │    └────────────┬───────────┘
           │               │                 │
           │               │    ┌────────────▼───────────┐
           │               │    │ OneCycleLR(             │
           │               │    │   max_lr=5e-4,          │
           │               │    │   steps_per_epoch=      │
           │               │    │   len(train_loader))    │
           │               │    └────────────┬───────────┘
           │               │                 │
           │               │    ┌────────────▼───────────┐
           │               │    │ sanity check:           │
           │               │    │ model(first_batch)      │
           │               │    │ 验证I/O shape            │
           │               │    └────────────┬───────────┘
           │               │                 │
           └───────────────┼─────────────────┘
                           │
                    ┌──────▼───────┐
                    │ epoch loop   │ ← for epoch in 1..epochs
                    │ 1..epochs    │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌────────────┐ ┌────────────┐ ┌──────────────┐
     │run_epoch   │ │run_epoch   │ │ 打印 epoch   │
     │(train)     │ │(val)       │ │ 指标          │
     │optimizer✓  │ │optimizer✗  │ │              │
     │scheduler✓  │ │scheduler✗  │ │              │
     └─────┬──────┘ └─────┬──────┘ └──────┬───────┘
           │              │                │
           └──────────────┼────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
     ┌────────────┐ ┌──────────┐ ┌──────────┐
     │ 保存       │ │val_mpjpe │ │val_pck   │
     │ last.pth   │ │ 改善?    │ │_0_2 改善? │
     └────────────┘ └────┬─────┘ └────┬─────┘
                    YES──┤      YES───┤
                    ▼         ▼
              ┌──────────┐ ┌──────────────┐
              │ 保存     │ │ 保存         │
              │ best_val │ │ best_val     │
              │ _mpjpe   │ │ _pck_0_2     │
              └──────────┘ └──────────────┘
                    │              │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ append_csv   │
                    │ → train_log  │
                    │   .csv       │
                    └──────────────┘
```

---

## 四、eval.py 深入剖析

### 4.1 评估架构概览

eval.py 提供三个层次的评估粒度：

| 层次 | 函数 | 输出 |
|------|------|------|
| **全局指标** | `evaluate_model()` | MPJPE, PCK@0.1~0.5 |
| **细粒度分解** | `collect_metric_breakdowns()` | 按关节/动作/环境拆分 |
| **可视化** | `save_visualizations()` | CSI 热图 + GT + 预测骨架 PNG |

### 4.2 评价指标计算

```python
# 躯干参考长度（用于 PCK 归一化）
torso_scale = ||keypoints[RIGHT_SHOULDER(2)] - keypoints[LEFT_HIP(11)]||

# 平均每关节位置误差
MPJPE = mean(||pred[i] - target[i]||)  for i = 0..17

# 正确关键点百分比（每个关节独立判断）
PCK@τ = mean(||pred[i] - target[i]|| < torso_scale × τ)
```

### 4.3 可视化逻辑详解

```
save_visualizations(model, loader, output_dir, device, max_visualizations)
         │
         │ 维护 visited = set() 用于 (action, environment) 去重
         │
         ▼
    ┌──────────────────────────────────────────────┐
    │  for batch in loader:                        │
    │    model_input, target = prepare_model_input  │
    │    predictions = model(model_input)           │
    │                                              │
    │    for i in range(batch_size):               │
    │      pair = (action[i], environment[i])       │
    │                                              │
    │      if pair already visited: continue        │ ← 去重
    │      if len(visited) >= max: break            │ ← 上限控制
    │                                              │
    │      ┌───────────────────────────────────┐   │
    │      │  创建 3 行 1 列的 matplotlib 图   │   │
    │      │                                  │   │
    │      │  [Panel 1] imshow                 │   │
    │      │  csi_amplitude: [3,114,64]       │   │
    │      │  → reshape 为 [342, 64]           │   │
    │      │  即 3天线×114子载波 × 64时间步     │   │
    │      │  cmap="jet"                       │   │
    │      │                                  │   │
    │      │  [Panel 2] scatter + plot         │   │
    │      │  Ground Truth 骨架 (绿色)          │   │
    │      │  18 关键点 + 19 骨骼连线           │   │
    │      │  ax.invert_yaxis()                │   │
    │      │                                  │   │
    │      │  [Panel 3] scatter + plot         │   │
    │      │  WiFlow 预测骨架 (红色)            │   │
    │      │  18 关键点 + 19 骨骼连线           │   │
    │      │  ax.invert_yaxis()                │   │
    │      │                                  │   │
    │      │  保存为:                           │   │
    │      │  {action}_{env}_frame{N}.png      │   │
    │      └───────────────────────────────────┘   │
    └──────────────────────────────────────────────┘
```

**Panel 1 的 `reshape(3*114, 64)` 设计意图：** CSI 数据形状为 `[3, 114, 64]`，直接展开为 `342 × 64` 的热图。Y 轴前 114 行对应天线 1、中间 114 行对应天线 2、后 114 行对应天线 3。在一个图中同时展示所有天线的时频特征，便于观察不同天线之间的信号模式差异。

### 4.4 评估结果文件结构

```
outputs/eval/
├── per_joint_metrics.csv         # 18 行 × (joint_index, sample_count, mpjpe, pck_0_2)
├── per_action_metrics.csv        # N 行 × (action, sample_count, mpjpe, pck_0_2)
├── per_environment_metrics.csv   # M 行 × (environment, sample_count, mpjpe, pck_0_2)
├── walk_lab_frame100.png         # 可视化: 步行在实验室环境
├── run_corridor_frame50.png      # 可视化: 跑步在走廊环境
└── ...
```

### 4.5 模型加载与还原

```python
def load_checkpoint_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_config = checkpoint["train_config"]   # 从 checkpoint 读取训练配置

    model = WiFlowModel(
        input_channels=train_config.get("input_channels", 3),
        axial_mode=train_config.get("axial_mode", "spatial_then_temporal"),
        decoder_type=train_config.get("decoder_type", "joint"),
        heatmap_size=train_config.get("heatmap_size", 36),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, input_channels
```

从 checkpoint 的 `train_config` 字段自动读取模型架构参数，确保评估时还原的模型与训练时完全一致，无需手动指定架构。

### 4.6 与训练过程的交互机制

```
train.py                              eval.py
────────                              ───────
run_training()                        main()
  │                                     │
  ├── WiFlowModel(...)                  ├── load_checkpoint_model()
  ├── AdamW(...)                          │
  ├── OneCycleLR(...)                     ├── torch.load(checkpoint)
  │                                       ├── 读取 train_config
  ├── for epoch in 1..50:                 │   ├── axial_mode
  │   ├── train_epoch()                   │   ├── decoder_type
  │   ├── val_epoch()                     │   ├── heatmap_size
  │   ├── save_checkpoint()               │   └── input_channels
  │   │   ├── model_state_dict     ───────┤
  │   │   ├── optimizer_state_dict        ├── WiFlowModel(...)
  │   │   ├── scheduler_state_dict        ├── model.load_state_dict()
  │   │   ├── train_config         ───────┤   → 完全还原训练时的模型
  │   │   └── best_metric                 │
  │   └── append_csv_row()                ├── evaluate_model()
  │                                       │   → 导入 train.py 的
  │                                       │     compute_metrics
  │                                       │     compute_torso_scale
  │                                       │     extract_prediction_keypoints
  │                                       │     prepare_model_input
  │                                       │     select_device
  │                                       │
  │                                       ├── collect_metric_breakdowns()
  │                                       │   → per_joint/per_action/
  │                                       │     per_environment
  │                                       │
  │                                       └── save_visualizations()
  │                                           → matplotlib PNG
  │
  └── train_log.csv (记录所有 epoch 指标)
```

**eval.py 从 train.py 导入的工具函数：**

| 导入函数 | 作用 |
|---------|------|
| `compute_metrics` | MPJPE + PCK@all 计算 |
| `compute_torso_scale` | 躯干归一化参考长度 |
| `extract_prediction_keypoints` | 统一处理 dict/Tensor 输出格式 |
| `prepare_model_input` | CSI + keypoints 设备转移 |
| `select_device` | CUDA 可用性检测与 fallback |

### 4.7 eval.py 核心流程图

```
                        ┌──────────┐
                        │  main()  │
                        └────┬─────┘
                             │
                    ┌────────┼────────┐
                    ▼        ▼        ▼
             ┌──────────┐ ┌──────┐ ┌──────────────┐
             │ 解析 CLI  │ │选设备 │ │ 加载 checkpoint│
             │ dataset   │ │cuda/ │ │ load_checkpoint│
             │ checkpoint│ │cpu   │ │ _model()      │
             │ output    │ └──┬───┘ └──────┬───────┘
             └────┬─────┘    │             │
                  │          │    ┌────────▼──────────┐
                  │          │    │ torch.load(path)   │
                  │          │    │ 读取 train_config   │
                  │          │    │   → axial_mode     │
                  │          │    │   → decoder_type   │
                  │          │    │   → heatmap_size   │
                  │          │    │   → input_channels │
                  │          │    │ WiFlowModel(...)   │
                  │          │    │ load_state_dict()  │
                  │          │    │ model.eval()       │
                  │          │    └────────┬──────────┘
                  │          │             │
                  │    ┌─────┴─────────────┴──────┐
                  │    │create_memmap_data_loader │
                  │    │  split="test"            │
                  │    │  shuffle=False           │
                  │    └─────────────┬────────────┘
                  │                  │
                  │    ┌─────────────┼──────────────┐
                  │    │             │              │
                  │    ▼             ▼              ▼
                  │  ┌──────────┐ ┌────────────┐ ┌──────────────┐
                  │  │evaluate  │ │collect_    │ │save_         │
                  │  │_model()  │ │metric_     │ │visualizations│
                  │  │          │ │breakdowns()│ │()            │
                  │  │→ mpjpe   │ │            │ │              │
                  │  │→ pck@τ   │ │→ per_joint │ │→ CSI heatmap │
                  │  └────┬─────┘ │  _metrics  │ │→ GT skeleton │
                  │       │       │  .csv      │ │→ Pred skeleton│
                  │       │       │→ per_action│ │→ PNG files   │
                  │       │       │  _metrics  │ └──────┬───────┘
                  │       │       │  .csv      │        │
                  │       │       │→ per_env   │        │
                  │       │       │  _metrics  │        │
                  │       │       │  .csv      │        │
                  │       │       └─────┬──────┘        │
                  │       │             │               │
                  │       ▼             ▼               ▼
                  │  ┌──────────────────────────────────────┐
                  │  │          print 结果汇总               │
                  │  │  Test Metrics + 文件保存路径          │
                  │  └──────────────────────────────────────┘
                  │
                  └──────────────────────────────┘
```

---

## 五、组件间依赖关系总览

```
                      ┌──────────────────┐
                      │   train.py       │──┐
                      │   eval.py        │  │
                      └────────┬─────────┘  │
                               │ imports    │ imports
              ┌────────────────┼────────────┼──────────────┐
              │                │            │              │
              ▼                ▼            ▼              │
    ┌─────────────┐  ┌──────────────┐  ┌──────────────┐   │
    │ dataloader  │  │ models/      │  │ pose_targets │   │
    │ .py         │  │ __init__.py  │  │ .py          │◄──┘
    └──────┬──────┘  └──────┬───────┘  └──────┬───────┘
           │                │                  │
           ▼                ▼                  ▼
    ┌─────────────┐  ┌──────────────────┐  ┌──────────────┐
    │ data/       │  │ models/skeleton  │◄─┤ models/      │
    │ memmap_     │  │ .py              │  │ skeleton.py  │
    │ dataset.py  │  │ (OPENPOSE_BONE   │  └──────────────┘
    └──────┬──────┘  │  _EDGES, etc.)   │
           │         └──────────────────┘
           ▼
    ┌─────────────┐
    │ data/       │
    │ heatmap_gt  │
    │ .py (NumPy) │  ← 仅被 memmap_dataset 使用（数据预处理）
    └─────────────┘

依赖规则:
  - train.py → 所有模块（训练需要完整流水线）
  - eval.py → dataloader + models + train.py 中的工具函数
  - models/__init__.py → 所有 models 子模块 + skeleton
  - dataloader.py → data/memmap_dataset.py
  - pose_targets.py → models/skeleton.py (Torch 版目标合成)
  - data/heatmap_gt.py → 独立 (NumPy 版目标合成，仅用于数据集构建)
```

---

## 六、关键设计决策

1. **单帧模型**：CSI 输入为 `[B, 3, 114, 64]`，3 天线 × 114 子载波 × 64 时间包（由原始 10 个时间采样点经 `scipy.signal.resample` 上采样至 64）。

2. **仅使用振幅**：仅使用 CSI 振幅（3 通道，每根天线一个），不使用相位信息。

3. **对称时频下采样**：`SymmetricResidualDownsampleBlock` 使用 stride=2 卷积同时下采样时间轴（64→32→16）和子载波轴（114→57→29），保留时间运动线索。

4. **轴向注意力**：分别在空间（29 个子载波位置）和时间（16 个时间步）轴上应用自注意力，避免对 29×16=464 个 token 进行全注意力所需的 O(N²) 代价。

5. **可学习关节查询**：18 个可学习的关节查询直接交叉注意力于 464 个时空 token，随后经过 GNN 骨架约束 + 关节自注意力 + MLP 坐标回归。无需中间池化层，查询直接关注特征图。

6. **三种解码器变体**：
   - `joint`（默认）：扁平的交叉注意力
   - `hierarchical`：由粗到精的分阶段解码
   - `heatmap_msfn`：类 MultiFormer 的多阶段热图回归，带 PAPM 反馈

7. **零拷贝数据加载**：`np.load(mmap_mode='r')` 提供操作系统缓存 I/O，无 HDF5 开销。三种预计算归一化变体减少训练时计算量。

8. **OpenPose18 骨架**：18 个关节，19 条骨骼边（以 neck 为中心关节，含眼睛/耳朵）。PCK 使用 right_shoulder（索引 2）到 left_hip（索引 11）作为躯干参考。

### 6.1 遗留/废弃测试文件

以下 4 个测试文件引用了已删除的类，将导致导入失败：

| 测试文件 | 缺失的导入 |
|---------|-----------|
| `tests/test_wiflow_skeleton_decoder.py` | `WiFlowSkeletonDecoder` |
| `tests/test_wiflow_temporal_encoder.py` | `WiFlowTemporalEncoder` |
| `tests/test_wiflow_spatial_temporal_fuser.py` | `WiFlowSpatialTemporalFuser` |
| `tests/test_wiflow_attention_pooler.py` | `WiFlowAttentionPooler` |

---

## 七、总结

| 维度 | train.py 特点 | eval.py 特点 |
|------|-------------|-------------|
| **解码器适配** | 双轨损失函数 (坐标 L1 / 热图 MSE) | `extract_prediction_keypoints` 统一处理 |
| **损失监督** | 坐标 + 骨骼长度 / PCM + PAF 多阶段 | 仅推理，无损失 |
| **优化策略** | AdamW + OneCycleLR (warmup + cos退火) | N/A |
| **正则化** | Weight Decay + Grad Clip + Bone Prior + LR Schedule | N/A |
| **模型选择** | 按 val_mpjpe / val_pck_0_2 保存最佳 checkpoint | 加载指定 checkpoint |
| **输出** | train_log.csv + .pth checkpoints | per_joint/action/env CSVs + PNG 可视化 |
| **与训练交互** | — | 读取 train_config 还原模型架构，复用训练工具函数 |
