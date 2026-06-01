# Tier 3–6 Progressive Unfreezing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `apply_finetune_tier()` from 2 tiers to 6, enabling progressive unfreezing of axial_encoder submodules.

**Architecture:** Single-function refactor in `train.py`. Replace the hardcoded tier-1/tier-2 branches with a keyword-driven loop shared across tiers 1–5, plus a full-unfreeze path for tier 6. No model changes, no new CLI arguments, fully backward-compatible.

**Tech Stack:** Python 3.10+, PyTorch

---

### File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `train.py:367-380` | Modify | Replace `apply_finetune_tier()` with 6-tier version |
| `train.py:528` | Modify | Update `--freeze-tier` help text |

---

### Task 1: Replace `apply_finetune_tier()` function

**Files:**
- Modify: `train.py:367-380`

- [ ] **Step 1: Apply the edit**

Replace the entire `apply_finetune_tier` function (lines 367–380) with the 6-tier version:

```python
def apply_finetune_tier(model: nn.Module, tier: int = 1) -> int:
    trainable_params = 0
    if tier == 1:
        keep_kw = ("norm", "bn", "ln", "joint_queries", "coordinate_head")
    elif tier == 2:
        keep_kw = ("norm", "bn", "ln", "joint_queries", "coordinate_head", "decoder.")
    elif tier == 3:
        keep_kw = ("norm", "bn", "ln", "joint_queries", "coordinate_head", "decoder.",
                   "spatial_attention.")
    elif tier == 4:
        keep_kw = ("norm", "bn", "ln", "joint_queries", "coordinate_head", "decoder.",
                   "spatial_attention.", "temporal_attention.")
    elif tier == 5:
        keep_kw = ("norm", "bn", "ln", "joint_queries", "coordinate_head", "decoder.",
                   "spatial_attention.", "temporal_attention.", "channel_projection")
    elif tier == 6:
        for param in model.parameters():
            param.requires_grad = True
            trainable_params += param.numel()
        total = sum(p.numel() for p in model.parameters())
        print(f"Freeze tier {tier}: {trainable_params}/{total} parameters trainable "
              f"({trainable_params / total * 100:.1f}%)")
        return trainable_params
    else:
        raise ValueError(f"Unknown freeze tier: {tier}. Valid tiers: 1-6")

    for name, param in model.named_parameters():
        keep = any(kw in name.lower() for kw in keep_kw)
        param.requires_grad = keep
        if keep:
            trainable_params += param.numel()

    total = sum(p.numel() for p in model.parameters())
    print(f"Freeze tier {tier}: {trainable_params}/{total} parameters trainable "
          f"({trainable_params / total * 100:.1f}%)")
    return trainable_params
```

---

### Task 2: Update `--freeze-tier` help text

**Files:**
- Modify: `train.py:528`

- [ ] **Step 1: Apply the edit**

Replace the help text on the `--freeze-tier` argument:

```
    parser.add_argument("--freeze-tier", type=int, default=1,
                        help="Freeze tier 1 (norms + head only) or 2 (+ decoder).")
```

→

```
    parser.add_argument("--freeze-tier", type=int, default=1,
                        help="Freeze tier: 1(norms+head) 2(+decoder) 3(+spatial_attn) "
                             "4(+temporal_attn) 5(+channel_proj) 6(full)")
```

---

### Task 3: Smoke-test with dry run

**Files:** None (read-only verification)

- [ ] **Step 1: Verify Tier 1 unchanged**

```bash
python -c "
import torch
import sys; sys.path.insert(0, '.')
from train import apply_finetune_tier
from models import WiFlowModel
m = WiFlowModel()
n = apply_finetune_tier(m, tier=1)
total = sum(p.numel() for p in m.parameters())
print(f'Tier 1: {n}/{total} trainable')
"
```

Expected: Output matches pre-change Tier 1 behavior (norms + joint_queries + coordinate_head only).

- [ ] **Step 2: Verify Tier 2 unchanged**

```bash
python -c "
import torch
import sys; sys.path.insert(0, '.')
from train import apply_finetune_tier
from models import WiFlowModel
m = WiFlowModel()
n = apply_finetune_tier(m, tier=2)
total = sum(p.numel() for p in m.parameters())
print(f'Tier 2: {n}/{total} trainable')
"
```

Expected: More trainable than Tier 1 (adds decoder), matches pre-change Tier 2.

- [ ] **Step 3: Verify Tier 3–5 monotonic increase**

```bash
python -c "
import torch
import sys; sys.path.insert(0, '.')
from train import apply_finetune_tier
from models import WiFlowModel
for t in range(1, 7):
    m = WiFlowModel()
    n = apply_finetune_tier(m, tier=t)
    print(f'Tier {t}: {n} trainable')
"
```

Expected: Tier 1 < Tier 2 < Tier 3 < Tier 4 < Tier 5 < Tier 6 = total.

- [ ] **Step 4: Verify Tier 6 is 100%**

```bash
python -c "
import torch
import sys; sys.path.insert(0, '.')
from train import apply_finetune_tier
from models import WiFlowModel
m = WiFlowModel()
n = apply_finetune_tier(m, tier=6)
total = sum(p.numel() for p in m.parameters())
assert n == total, f'Tier 6 should be {total} but got {n}'
print('Tier 6: PASS')
"
```

Expected: `Tier 6: PASS`.

---

### Task 4: Commit

- [ ] **Step 1: Commit the changes**

```bash
git add train.py
git commit -m "feat: extend apply_finetune_tier to 6 levels (Tier 3-6 progressive unfreezing)"
```

- [ ] **Step 2: Push**

```bash
git push
```
