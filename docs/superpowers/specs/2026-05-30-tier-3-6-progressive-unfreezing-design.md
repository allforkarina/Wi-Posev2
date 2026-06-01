# Tier 3–6 Progressive Unfreezing for Cross-Domain Few-Shot Finetune

**Date**: 2026-05-30
**Status**: Approved

## Motivation

当前 `train.py` 的 `apply_finetune_tier()` 只支持 Tier 1（norms + head）和 Tier 2（+ decoder），Tier 3 及以上直接报错。消融实验需要逐级解冻 axial_encoder 以回答"时空注意力模式是否跨域通用"。

## Design

### 单一改动：`train.py` — `apply_finetune_tier()`

替换现有函数，扩展为 6 级：

| Tier | 解冻规则 | 关键词 | 要回答的问题 |
|------|---------|--------|------------|
| 1 | norms + joint_queries + coordinate_head | `norm`, `bn`, `ln`, `joint_queries`, `coordinate_head` | 分布统计量适配够吗？ |
| 2 | + decoder | `decoder.` | 解码器需要域特定知识吗？ |
| 3 | + spatial_attention | `spatial_attention.` | 空间位置注意力跨域通用吗？ |
| 4 | + temporal_attention | `temporal_attention.` | 时间序列注意力跨域通用吗？ |
| 5 | + channel_projection | `channel_projection` | 通道映射需要域特定知识吗？ |
| 6 | 全模型 | （跳过匹配循环） | 端到端适应上限在哪？ |

### 实现要点

- Tier 1–5 共用统一匹配循环，每级扩展 `keep_kw` 元组
- Tier 6 直接 `param.requires_grad = True` 遍历全部参数，避免遗漏新增参数
- `channel_projection` 子串同时命中 `channel_projection` 和 `concat_projection`（parallel_concat 模式），两者语义一致
- 错误信息更新为 `"Valid tiers: 1-6"`
- CLI `--freeze-tier` help 文本更新

### 兼容性

- **Tier 1/2 完全向后兼容**：关键词集合不变，匹配逻辑不变
- **旧 source_only checkpoint 可直接用于新 Tier**：`_run_finetune` 从 checkpoint 的 `train_config` 还原模型架构再加载权重，不依赖 tier
- **无新增 CLI 参数**：`--freeze-tier` 已是 `type=int`，自然接受 1–6

## Verification

| 检查项 | 方法 |
|--------|------|
| Tier 1/2 行为不变 | `--freeze-tier 1/2` 打印的 trainable/total 与改动前一致 |
| Tier 3–5 仅解冻目标模块 | 打印的百分比逐级递增，总参数数不变 |
| Tier 6 全解冻 | 打印 `X/X parameters trainable (100.0%)` |
| 训练可收敛 | Tier 3 跑 30 epochs，loss 下降 |
