# eval.py env/split_mode 支持 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** eval.py 从 checkpoint 的 `train_config` 自动读取 `envs`/`split_mode`，使评估时的数据加载参数与训练时完全一致，避免环境不匹配导致的指标失真。

**架构：** 方案 B（零配置）— 修改 `load_checkpoint_model` 返回 `train_config` dict，在 `main()` 中提取 `envs`/`split_mode` 并透传给 `create_memmap_data_loader`。`train.py` 无需修改（`asdict(TrainConfig)` 已序列化这两个字段）。

**技术栈：** Python 3.10+, pathlib, PyTorch

**影响范围：** 仅 `eval.py`，两处修改，约 10 行净增代码。

---

### Task 1: 修改 `load_checkpoint_model` 返回值，携带 `train_config`

**文件：**
- Modify: `eval.py:L22` (函数签名), `eval.py:L42` (return 语句), `eval.py:L275` (调用点)

- [ ] **Step 1: 修改函数签名与返回**

将 `load_checkpoint_model` 的返回值从 `tuple[WiFlowModel, int]` 改为 `tuple[WiFlowModel, int, dict]`，新增返回 `train_config`。

```python
# eval.py:L19-L43 — 修改前
def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[WiFlowModel, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing model_state_dict: {checkpoint_path}")
    train_config = checkpoint.get("train_config")
    if not isinstance(train_config, Mapping):
        raise KeyError(f"Checkpoint is missing train_config: {checkpoint_path}")

    axial_mode = str(train_config.get("axial_mode", "spatial_then_temporal"))
    decoder_type = str(train_config.get("decoder_type", "joint"))
    heatmap_size = int(train_config.get("heatmap_size", 36))
    input_channels = int(train_config.get("input_channels", 3))
    model = WiFlowModel(
        input_channels=input_channels,
        axial_mode=axial_mode,
        decoder_type=decoder_type,
        heatmap_size=heatmap_size,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, input_channels

# eval.py:L19-L43 — 修改后
def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[WiFlowModel, int, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing model_state_dict: {checkpoint_path}")
    train_config = checkpoint.get("train_config")
    if not isinstance(train_config, Mapping):
        raise KeyError(f"Checkpoint is missing train_config: {checkpoint_path}")

    axial_mode = str(train_config.get("axial_mode", "spatial_then_temporal"))
    decoder_type = str(train_config.get("decoder_type", "joint"))
    heatmap_size = int(train_config.get("heatmap_size", 36))
    input_channels = int(train_config.get("input_channels", 3))
    model = WiFlowModel(
        input_channels=input_channels,
        axial_mode=axial_mode,
        decoder_type=decoder_type,
        heatmap_size=heatmap_size,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, input_channels, dict(train_config)
```

**关键变更：**
- 返回类型 `tuple[WiFlowModel, int]` → `tuple[WiFlowModel, int, dict]`
- `return model, input_channels` → `return model, input_channels, dict(train_config)`

- [ ] **Step 2: 更新 `main()` 调用点**

```python
# eval.py:L275 — 修改前
model, input_channels = load_checkpoint_model(args.checkpoint, device)

# eval.py:L275 — 修改后
model, input_channels, train_config = load_checkpoint_model(args.checkpoint, device)
```

- [ ] **Step 3: 验证语法**

```bash
conda activate WiFiPose && python -c "import ast; ast.parse(open('eval.py', encoding='utf-8').read()); print('Syntax OK')"
```

预期：`Syntax OK`

- [ ] **Step 4: 提交**

```bash
git add eval.py
git commit -m "feat: load_checkpoint_model returns train_config dict"
```

---

### Task 2: 在 `main()` 中使用 `train_config` 传递 env/split_mode

**文件：**
- Modify: `eval.py:L276-L282` (test_loader 创建处)

- [ ] **Step 1: 提取 env 和 split_mode，透传给 DataLoader**

```python
# eval.py:L276-L282 — 修改前
    test_loader = create_memmap_data_loader(
        data_dir=args.dataset_root,
        split="test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

# eval.py:L276-L284 — 修改后
    train_envs = train_config.get("envs")
    train_split_mode = train_config.get("split_mode", "subject")

    test_loader = create_memmap_data_loader(
        data_dir=args.dataset_root,
        split="test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        envs=train_envs,
        split_mode=str(train_split_mode),
    )
```

**关键点：**
- `train_config.get("envs")` — 旧 checkpoint 无此 key 时返回 `None`（全部环境），语义正确
- `train_config.get("split_mode", "subject")` — 旧 checkpoint 无此 key 时默认 `"subject"`，语义正确
- `str(train_split_mode)` — 防御性转换（`asdict` 序列化为字符串）

- [ ] **Step 2: 验证语法**

```bash
conda activate WiFiPose && python -c "import ast; ast.parse(open('eval.py', encoding='utf-8').read()); print('Syntax OK')"
```

预期：`Syntax OK`

- [ ] **Step 3: 提交**

```bash
git add eval.py
git commit -m "feat: eval.py auto-reads env/split_mode from checkpoint train_config"
```

---

### Task 3: 端到端验证

- [ ] **Step 1: 运行已有测试确保无回归**

```bash
conda activate WiFiPose && python -m pytest tests/test_memmap_dataset.py tests/test_dataloader.py -v
```

预期：20 passed

- [ ] **Step 2: 模拟 checkpoint 兼容性测试**

```python
# 在 Python 中验证新旧 checkpoint 兼容性
python -c "
import torch
# 模拟旧 checkpoint（无 envs/split_mode 字段）
old_config = {'axial_mode': 'spatial_then_temporal', 'decoder_type': 'joint', 'heatmap_size': 36, 'input_channels': 3}
assert old_config.get('envs') is None
assert old_config.get('split_mode', 'subject') == 'subject'

# 模拟新 checkpoint（有 envs/split_mode）
new_config = {'axial_mode': 'spatial_then_temporal', 'envs': ['env1'], 'split_mode': 'subject'}
assert new_config.get('envs') == ['env1']
assert new_config.get('split_mode', 'subject') == 'subject'
print('Backward compatibility OK')
"
```

预期：`Backward compatibility OK`

- [ ] **Step 3: 更新 AGENTS.md**

在 eval 命令说明中注明 env/split_mode 自动从 checkpoint 读取：

```markdown
Evaluate one checkpoint:

```bash
python eval.py --dataset-root data\mmfi_pose --checkpoint outputs\train\best_val_mpjpe.pth --output-dir outputs\eval
```

The evaluation script automatically reads `envs` and `split_mode` from the checkpoint's `train_config`.
No manual --env or --split-mode flags are needed.
```

- [ ] **Step 4: 最终提交**

```bash
git add AGENTS.md
git commit -m "docs: update eval usage note for auto env/split_mode"
```

---

## 自审

**1. 需求覆盖：** eval.py 的 env/split_mode 缺口 → Task 1+2 完全覆盖。向后兼容 → Task 2 的 `.get()` 默认值处理。回归测试 → Task 3 Step 1。文档 → Task 3 Step 3。

**2. 无占位符：** 所有代码块均有具体实现。

**3. 类型一致性：** `train_config` 提取的 `envs` 类型为 `list[str] | None`，与 `create_memmap_data_loader(envs=...)` 参数签名 `Iterable[str] | None` 一致。