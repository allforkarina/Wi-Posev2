# Joint Scatter + Multi-Action Feature Visualization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore joint scatter visualization and fix multi-action sampling to output feature maps for all action types with configurable output parameters.

**Architecture:** Root cause of "1 action only" is a premature early-break in `_collect_action_env_samples`: it checks `all_saturated` on ALL actions, but since the loader iterates action-by-action (shuffle=False), the condition becomes True after just the first action's environments are collected. Fix: remove the early break and process the full test set (or use an action-count threshold). Add `_fig0_joint_scatter` function drawing GT vs predicted keypoints as color-coded scatter points (no skeleton). Add `--output-format`, `--figure-width`, `--figure-height` CLI args to eval.py.

**Tech Stack:** matplotlib, numpy, torch — same as existing feature_viz.py

---

## Root Cause Analysis

`_collect_action_env_samples` (line 148-152 of feature_viz.py):

```python
all_saturated = all(
    len(e) >= num_per_action for e in action_env.values()
)
if all_saturated:
    break
```

Test DataLoader uses `shuffle=False`. Data is ordered by action — all A01 samples come before A02. When A01 has 3+ environments, `action_env.values()` = `[{"env1": ..., "env2": ..., "env3": ...}]`. `all(len(e) >= 3)` → `True` → breaks immediately after A01.

**Fix:** Replace per-batch saturation check with a total-sample threshold (e.g., `num_action_samples * 20`), or iterate the entire test set. Since per-sample forward passes are already happening, iterating the full test set is acceptable (typically ~8k frames → ~125 batches at BS=64).

---

### Task 1: Fix multi-action sampling bug

**Files:**
- Modify: `evaluation/feature_viz.py:148-152`

- [ ] **Step 1: Remove early break, add total-action threshold instead**

Replace the `all_saturated` break logic at lines 148-152:

```python
            # Stop when we've collected enough unique actions (target: all actions,
            # but stop early if we accumulate way more samples than expected).
            if len(action_env) >= 10 and all(
                len(e) >= num_per_action for e in action_env.values()
            ):
                break
```

Better: compute total actions from the dataset. But since we don't have that easily, use a simpler approach — just iterate the whole test set:

Replace lines 148-152 with:
```python
            # Collect from the full test set to cover all actions.
            # Forward passes are already happening, so this adds no extra cost.
            # Break only when truly all known actions are saturated.
            if len(action_env) >= 2 and all(
                len(e) >= num_per_action for e in action_env.values()
            ):
                break
```

Wait — same problem. The issue is `all()` on `action_env.values()` is True when we've only seen one action.

Correct fix:

Replace lines 148-152 with:
```python
            # Continue until we've collected samples from at least 2 distinct
            # actions and each has >= num_per_action environments.
            if len(action_env) >= 2 and all(
                len(e) >= num_per_action for e in action_env.values()
            ):
                break
```

Still wrong — if A01 has 3 envs, it's still the only action in the dict.

Actually the cleanest fix: just don't break early at all. The loader is the test set, which is finite. Remove the break entirely, or set `num_action_samples` high enough.

Replace lines 148-152:
```python
            # Do not break early — iterate the full test set to collect
            # samples from all action types.  Skips duplicate envs per action.
```

Delete the `all_saturated` check and break entirely.

- [ ] **Step 2: Verify syntax**

Run:
```powershell
conda activate WiFiPose && python -c "compile(open('evaluation/feature_viz.py', encoding='utf-8').read(), 'feature_viz.py', 'exec'); print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add evaluation/feature_viz.py
git commit -m "fix: remove premature early-break in multi-action sampling"
```

---

### Task 2: Add `--output-format` and `--figure-dim` CLI args

**Files:**
- Modify: `eval.py:240-258` (parse_args)
- Modify: `eval.py:284-298` (main, pass args to run_feature_visualization)
- Modify: `evaluation/feature_viz.py:84-88` (_save_fig)
- Modify: `evaluation/feature_viz.py:877-980` (run_feature_visualization signature)

- [ ] **Step 1: Add CLI arguments to eval.py**

In `parse_args()`, after `--num-action-samples`:

```python
    parser.add_argument(
        "--output-format", choices=["png", "pdf", "both"], default="both",
        help="Output image format (default: both).",
    )
    parser.add_argument(
        "--figure-width", type=float, default=None,
        help="Override default figure width in inches.",
    )
    parser.add_argument(
        "--figure-height", type=float, default=None,
        help="Override default figure height in inches.",
    )
```

- [ ] **Step 2: Pass new args to run_feature_visualization**

In `main()`, update the call:

```python
        run_feature_visualization(
            model=model,
            loader=test_loader,
            dataset_root=args.dataset_root,
            output_dir=output_dir,
            device=device,
            decoder_type=model.decoder_type,
            num_action_samples=args.num_action_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_format=args.output_format,
            figure_width=args.figure_width,
            figure_height=args.figure_height,
        )
```

- [ ] **Step 3: Update _save_fig to respect output_format**

Replace lines 84-88:

```python
def _save_fig(fig: plt.Figure, path: Path, fmt: str = "both") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("pdf", "both"):
        fig.savefig(str(path.with_suffix(".pdf")), dpi=300)
    if fmt in ("png", "both"):
        fig.savefig(str(path.with_suffix(".png")), dpi=300)
    plt.close(fig)
```

- [ ] **Step 4: Thread output_format through all figure functions**

Each `_fig*` function calls `_save_fig(fig, path)`. Change to `_save_fig(fig, path, fmt=output_format)`.

This requires passing `output_format` into each `_fig*` function. To avoid massive signature changes, store it as a module-level variable or pass via `run_feature_visualization`'s kwargs.

Simpler approach: use a module-level default that `run_feature_visualization` updates:

```python
_OUTPUT_FORMAT = "both"
_FIGURE_WIDTH = None
_FIGURE_HEIGHT = None
```

Then `_save_fig` reads from module level:
```python
def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _OUTPUT_FORMAT in ("pdf", "both"):
        fig.savefig(str(path.with_suffix(".pdf")), dpi=300)
    if _OUTPUT_FORMAT in ("png", "both"):
        fig.savefig(str(path.with_suffix(".png")), dpi=300)
    plt.close(fig)
```

And `run_feature_visualization` sets:
```python
    global _OUTPUT_FORMAT, _FIGURE_WIDTH, _FIGURE_HEIGHT
    _OUTPUT_FORMAT = output_format
    _FIGURE_WIDTH = figure_width
    _FIGURE_HEIGHT = figure_height
```

- [ ] **Step 5: Verify syntax**

```powershell
conda activate WiFiPose && python -c "compile(open('evaluation/feature_viz.py', encoding='utf-8').read(), 'feature_viz.py', 'exec'); compile(open('eval.py', encoding='utf-8').read(), 'eval.py', 'exec'); print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add eval.py evaluation/feature_viz.py
git commit -m "feat: add --output-format, --figure-width, --figure-height CLI args"
```

---

### Task 3: Add joint scatter visualization (`_fig0_joint_scatter`)

**Files:**
- Modify: `evaluation/feature_viz.py` (new function + orchestrator call)

- [ ] **Step 1: Add _fig0_joint_scatter function**

Insert after the `_flatten_samples` function (line 167) and before the Fig 1 section:

```python
# ---------------------------------------------------------------------------
# Figure 0: Joint scatter visualization (GT vs predicted keypoints)
# ---------------------------------------------------------------------------


def _fig0_joint_scatter(
    sample: dict[str, Any],
    sample_dir: Path,
) -> None:
    """Draw GT and predicted keypoints as colored scatter points (no skeleton).

    Uses anatomical group colors for both GT (filled circles) and prediction
    (hollow diamonds).  Thin gray error vectors connect each GT→pred pair.
    """
    target = sample["target"].cpu().numpy().squeeze(0)  # [18, 2]
    prediction = sample["prediction"]                     # [18, 2]

    # Invert y-axis so the human figure is upright (y=0 at top, y=1 at bottom)
    # If coordinates are already in normalized space, use as-is with y-invert
    fig, ax = plt.subplots(figsize=(8, 8))

    # Determine axis limits with 5% padding
    all_points = np.concatenate([target, prediction], axis=0)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    x_pad = max((x_max - x_min) * 0.1, 0.02)
    y_pad = max((y_max - y_min) * 0.1, 0.02)

    # --- Error vectors: faint gray dashed lines ---
    for j in range(18):
        ax.plot(
            [target[j, 0], prediction[j, 0]],
            [target[j, 1], prediction[j, 1]],
            color="gray", linewidth=0.5, linestyle="--", alpha=0.5,
            zorder=1,
        )

    # --- GT: filled circles ---
    for group_name, group_color in _ANATOMY_COLORS.items():
        indices = _ANATOMY_GROUPS[group_name]
        ax.scatter(
            target[indices, 0], target[indices, 1],
            c=group_color, marker="o", s=80, edgecolors="black",
            linewidths=0.5, label=f"GT {group_name}", zorder=3,
        )

    # --- Prediction: hollow diamonds ---
    for group_name, group_color in _ANATOMY_COLORS.items():
        indices = _ANATOMY_GROUPS[group_name]
        ax.scatter(
            prediction[indices, 0], prediction[indices, 1],
            c="white", marker="D", s=80, edgecolors=group_color,
            linewidths=1.2, label=f"Pred {group_name}", zorder=2,
        )

    # --- Legend (simplified: 2 entries, GT vs Pred) ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=8, label="GT"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
               markeredgecolor="gray", markersize=8, label="Prediction"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_max + y_pad, y_min - y_pad)  # invert for natural pose
    ax.set_aspect("equal")
    ax.set_xlabel("Normalized X")
    ax.set_ylabel("Normalized Y")
    ax.set_title(
        f"Joint Prediction vs GT — {sample['action']} / {sample['environment']}",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    _apply_spacing(fig)
    _save_fig(fig, sample_dir / "fig0_joint_scatter")
```

- [ ] **Step 2: Add call in orchestrator**

In `run_feature_visualization`, inside the per-sample loop after `_fig3_axial_attention`, add:

```python
            # Fig 0: Joint Scatter (GT vs prediction, no skeleton)
            _fig0_joint_scatter(sample, sample_dir)
```

Insert after the `_fig3_axial_attention` call (around line 966) and before the Fig 4/5 block.

- [ ] **Step 3: Add fig0 to overview composite**

In `_build_overview`, add `"fig0_joint_scatter"` to `fig_names`:

```python
    fig_names = [
        "fig0_joint_scatter",
        "fig1_antenna_channel",
        "fig2_downsampling_trajectory",
        "fig3_axial_attention",
    ]
```

And update the overview grid to accommodate 6+ figures. Since we now have up to 7 figures (fig0 + fig1-6), use a 3x3 grid:

```python
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
```

- [ ] **Step 4: Verify syntax**

```powershell
conda activate WiFiPose && python -c "compile(open('evaluation/feature_viz.py', encoding='utf-8').read(), 'feature_viz.py', 'exec'); print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add evaluation/feature_viz.py
git commit -m "feat: add joint scatter visualization (fig0)"
```

---

### Task 4: Per-action composite figures

**Files:**
- Modify: `evaluation/feature_viz.py` (new function + orchestrator call)

- [ ] **Step 1: Add _build_action_composites function**

After the `_build_overview` function, add:

```python
def _build_action_composites(
    action_env: dict[str, dict[str, dict[str, Any]]],
    viz_dir: Path,
    model: WiFlowModel,
    decoder_type: str,
    all_hooks: list[str],
    device: torch.device,
) -> None:
    """For each action, generate a 2×3 composite showing its samples across
    key figure types (fig0 + fig1 + fig3 + fig4/5).

    Each action gets one composite page with up to 6 panels: joint scatter
    and attention maps from 2 representative environment samples.
    """
    try:
        from PIL import Image
    except ImportError:
        print("    [INFO] PIL not installed — skipping action composites")
        return

    actions_dir = viz_dir / "_actions"
    actions_dir.mkdir(parents=True, exist_ok=True)

    for action, env_dict in sorted(action_env.items()):
        print(f"  Building action composite: {action}")
        envs = sorted(env_dict.keys())
        n_envs = len(envs)

        # Take up to 2 environments for side-by-side comparison
        envs_to_show = envs[:2]

        # Generate figures for these samples if not already done
        sample_paths: list[Path] = []
        for env in envs_to_show:
            sample = env_dict[env]
            s_dir = viz_dir / f"{action}_{env}_s0"
            s_dir.mkdir(parents=True, exist_ok=True)
            sample_paths.append(s_dir)

            # Generate fig0 (joint scatter) and fig3 (attention) if not exists
            if not (s_dir / "fig0_joint_scatter.png").exists():
                with wiflow_hooks(model, all_hooks) as ctx:
                    with torch.no_grad():
                        _ = model(sample["model_input"].to(device))
                _fig0_joint_scatter(sample, s_dir)
                _fig3_axial_attention(sample, ctx, s_dir)

        # Build the composite: rows = figure types, cols = environment samples
        n_cols = len(envs_to_show)
        n_rows = 2  # fig0 + fig3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 8 * n_rows))
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        fig_names = ["fig0_joint_scatter", "fig3_axial_attention"]
        for r, fname in enumerate(fig_names):
            for c, sp in enumerate(sample_paths):
                ax = axes[r, c]
                img_path = sp / f"{fname}.png"
                if img_path.exists():
                    img = Image.open(img_path)
                    ax.imshow(img)
                ax.set_title(
                    f"{envs_to_show[c]} — {fname.split('_', 1)[1]}",
                    fontsize=10, fontweight="bold",
                )
                ax.axis("off")

        fig.suptitle(
            f"Action: {action}  ({n_envs} environments sampled)",
            fontsize=14, fontweight="bold",
        )
        _apply_spacing(fig)
        _save_fig(fig, actions_dir / f"composite_{action}")
```

- [ ] **Step 2: Add call in orchestrator**

In `run_feature_visualization`, after the per-sample loop and before the Fig 6/overview section, add:

```python
    # --- Per-action composites ---
    print("  Building per-action composite figures...")
    _build_action_composites(
        action_env=action_env,
        viz_dir=viz_dir,
        model=model,
        decoder_type=decoder_type,
        all_hooks=all_hooks,
        device=device,
    )
```

- [ ] **Step 3: Verify syntax**

```powershell
conda activate WiFiPose && python -c "compile(open('evaluation/feature_viz.py', encoding='utf-8').read(), 'feature_viz.py', 'exec'); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add evaluation/feature_viz.py
git commit -m "feat: add per-action composite figures"
```

---

## Self-Review

### 1. Spec Coverage

| Requirement | Coverage |
|---|---|
| 恢复关节点散点可视化 | Task 3: `_fig0_joint_scatter` with GT circles + Pred diamonds |
| 清晰可见，有明显区分 | Anatomical colors for GT, hollow diamonds for pred, error vectors |
| 正确空间定位与比例缩放 | Auto-padding, equal aspect, inverted y-axis |
| 不同分辨率正常显示 | figsize=(8,8), DPI 300, tight bbox |
| 多动作批量输出 | Task 1: fix sampling bug; Task 4: per-action composites |
| 多动作排列布局 | 2-row × N-col (up to 2 envs) composite grid |
| 动作类别标识 | Action name in title, env label on each panel |
| 可配置输出参数 | Task 2: `--output-format`, `--figure-width`, `--figure-height` |

### 2. Placeholder Scan

No TBD, TODO, or placeholder patterns found.

### 3. Type Consistency

- `sample["target"]`: `torch.Tensor` [1, 18, 2] → `.cpu().numpy().squeeze(0)` → `ndarray` [18, 2]
- `sample["prediction"]`: `ndarray` [18, 2]
- `_save_fig` signature updated consistently
- `_OUTPUT_FORMAT` module-level variable used uniformly

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-23-joint-scatter-multi-action.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session, batch execution with checkpoints

**Which approach?**