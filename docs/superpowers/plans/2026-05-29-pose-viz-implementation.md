# Pose Joint Scatter Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--pose-viz` flag to `eval.py` that generates per-subject joint scatter plots (18 joints, distinct colors, GT vs Prediction, no skeleton) using fixed middle-frame sampling.

**Architecture:** New standalone module `evaluation/pose_viz.py` handles sampling + drawing. `eval.py` gains a `--pose-viz` flag wired to the new module. `evaluation/feature_viz.py` is surgically stripped of the old `_fig0_joint_scatter` function and its call sites.

**Tech Stack:** matplotlib, numpy, torch, pathlib

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `evaluation/feature_viz.py` | Modify | Remove `_fig0_joint_scatter` function and its 4 call sites |
| `evaluation/pose_viz.py` | Create | Middle-frame sampler, scatter plot drawer, orchestrator |
| `eval.py` | Modify | Add `--pose-viz` flag, mutual exclusion, wire to orchestrator |

---

### Task 1: Strip `_fig0_joint_scatter` from feature_viz.py

**Files:**
- Modify: `evaluation/feature_viz.py`

- [ ] **Step 1: Remove `_fig0_joint_scatter` function definition**

Delete lines 175–249 (the section comment through the function body).

In `evaluation/feature_viz.py`, remove:
```python
# Figure 0: Joint scatter visualization (GT vs predicted keypoints)
# ---------------------------------------------------------------------------


def _fig0_joint_scatter(
    sample: dict[str, Any],
    sample_dir: Path,
) -> None:
    """Draw GT and predicted keypoints as colored scatter points (no skeleton).
    ...
    _save_fig(fig, sample_dir / "fig0_joint_scatter")
```

- [ ] **Step 2: Remove the `_fig0_joint_scatter` call from `run_feature_visualization`**

Delete these two lines (currently ~lines 1076–1077):
```python
            # Fig 0: Joint Scatter (GT vs prediction, no skeleton)
            _fig0_joint_scatter(sample, sample_dir)
```

- [ ] **Step 3: Remove fig0 regen logic from `_build_action_composites`**

Replace the conditional regen block (currently ~lines 916–922):
```python
            if not (s_dir / "fig0_joint_scatter.png").exists() or not (s_dir / "fig3_axial_attention.png").exists():
                with wiflow_hooks(model, all_hooks) as ctx:
                    with torch.no_grad():
                        _ = model(sample["model_input"].to(device))
                if not (s_dir / "fig0_joint_scatter.png").exists():
                    _fig0_joint_scatter(sample, s_dir)
                if not (s_dir / "fig3_axial_attention.png").exists():
                    _fig3_axial_attention(sample, ctx, s_dir)
```

Replace with:
```python
            if not (s_dir / "fig3_axial_attention.png").exists():
                with wiflow_hooks(model, all_hooks) as ctx:
                    with torch.no_grad():
                        _ = model(sample["model_input"].to(device))
                if not (s_dir / "fig3_axial_attention.png").exists():
                    _fig3_axial_attention(sample, ctx, s_dir)
```

- [ ] **Step 4: Adjust `_build_action_composites` grid from 2-row to 1-row**

Change `n_rows = 2` to `n_rows = 1` (~line 927):
```python
        n_rows = 1  # fig3 only (was fig0 + fig3)
```

Change `fig_types` to only contain fig3 (~lines 938–941):
```python
        fig_types = [
            ("fig3_axial_attention", "Axial Attention"),
        ]
```

- [ ] **Step 5: Remove "fig0_joint_scatter" from `_build_overview` fig_names list**

Delete the `"fig0_joint_scatter",` line from the list (~line 799):
```python
    fig_names = [
        "fig1_antenna_channel",
        "fig2_downsampling_trajectory",
        "fig3_axial_attention",
        "fig4_joint_query_trajectory",
        "fig6_feature_pose_correlation",
    ]
```

- [ ] **Step 6: Verify feature_viz.py is clean**

Run: `grep -n "fig0\|_fig0_joint_scatter" evaluation/feature_viz.py`
Expected: No output (no remaining references).

- [ ] **Step 7: Commit**

```bash
git add evaluation/feature_viz.py
git commit -m "refactor: remove _fig0_joint_scatter from feature_viz, superseded by --pose-viz"
```

---

### Task 2: Create evaluation/pose_viz.py

**Files:**
- Create: `evaluation/pose_viz.py`

- [ ] **Step 1: Write the new module**

```python
"""Pose joint scatter visualization for eval.py --pose-viz.

Generates per-subject scatter plots comparing GT and predicted OpenPose18
keypoints. Each joint gets a distinct color; no skeleton bones are drawn.
Samples the middle frame (index 148) of each (action, subject) pair.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, Subset

from data.memmap_dataset import MemmapDataset
from dataloader import memmap_collate_fn
from train import extract_prediction_keypoints, prepare_model_input

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

FONT_FAMILY = "DejaVu Sans"

JOINT_COLORS = [
    "#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231", "#911EB4",
    "#46F0F0", "#F032E6", "#BCF60C", "#FABEBE", "#008080", "#E6BEFF",
    "#9A6324", "#FFFAC8", "#800000", "#AAFFC3", "#808000", "#FFD8B1",
]

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "image.interpolation": "nearest",
})

# ---------------------------------------------------------------------------
# Middle-frame sampling
# ---------------------------------------------------------------------------

MIDDLE_FRAME_OFFSET = 148  # 297 // 2


def _collect_middle_frames(dataset: MemmapDataset) -> list[int]:
    """Return dataset positions of the middle frame for each (action, subject).

    Groups all positions in *dataset* by (action, subject), sorts each group,
    and selects the 148th position (0-indexed). Returns a sorted list of
    positions suitable for ``Subset(dataset, result)``.
    """
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for pos in range(len(dataset)):
        abs_idx = int(dataset.indices[pos])
        action = str(dataset._actions[abs_idx])
        subject = str(dataset._samples[abs_idx])
        groups[(action, subject)].append(pos)

    selected: list[int] = []
    for (action, subject), positions in sorted(groups.items()):
        positions.sort()
        if len(positions) <= MIDDLE_FRAME_OFFSET:
            print(f"  [WARN] ({action}, {subject}) has only {len(positions)} "
                  f"frames, expected > {MIDDLE_FRAME_OFFSET} — skipping")
            continue
        selected.append(positions[MIDDLE_FRAME_OFFSET])

    return sorted(selected)


# ---------------------------------------------------------------------------
# Scatter plot drawing
# ---------------------------------------------------------------------------

def _draw_pose_scatter(
    target: np.ndarray,       # [18, 2]
    prediction: np.ndarray,   # [18, 2]
    action: str,
    subject: str,
    environment: str,
    output_dir: Path,
    output_format: str,
    figure_width: float | None,
    figure_height: float | None,
) -> None:
    """Draw and save a single joint scatter plot (Clean Modern style)."""
    fig_w = figure_width or 8.0
    fig_h = figure_height or 8.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # --- axis limits with 10% padding ---
    all_points = np.concatenate([target, prediction], axis=0)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    x_pad = max((x_max - x_min) * 0.1, 0.02)
    y_pad = max((y_max - y_min) * 0.1, 0.02)

    # --- background grid ---
    ax.set_facecolor("#ffffff")
    ax.grid(True, alpha=0.3, color="#e8e8e8", linewidth=0.5)

    # --- error vectors: thin gray dashed ---
    for j in range(18):
        ax.plot(
            [target[j, 0], prediction[j, 0]],
            [target[j, 1], prediction[j, 1]],
            color="#cccccc", linewidth=0.5, linestyle="--", zorder=1,
        )

    # --- GT: dashed hollow circles ---
    for j in range(18):
        ax.scatter(
            target[j, 0], target[j, 1],
            facecolors="none",
            edgecolors=JOINT_COLORS[j],
            marker="o", s=80, linewidths=1.8,
            linestyle="--", zorder=3,
        )

    # --- Prediction: filled circles with dark border ---
    for j in range(18):
        ax.scatter(
            prediction[j, 0], prediction[j, 1],
            facecolors=JOINT_COLORS[j],
            edgecolors="#333333",
            marker="o", s=80, linewidths=1.0, zorder=2,
        )
        # joint index label
        ax.annotate(
            str(j),
            (prediction[j, 0], prediction[j, 1]),
            fontsize=8, fontweight="bold",
            xytext=(8, -8), textcoords="offset points",
        )

    # --- legend ---
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="#888888", markersize=8, markeredgewidth=1.8,
               linestyle="--", label="GT"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markeredgecolor="#333333", markersize=8, label="Prediction"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_max + y_pad, y_min - y_pad)  # invert for natural pose
    ax.set_aspect("equal")
    ax.set_xlabel("Normalized X")
    ax.set_ylabel("Normalized Y")
    ax.set_title(
        f"Joint Prediction vs GT — {action} / {subject} / {environment}",
        fontsize=12, fontweight="bold",
    )

    fig.subplots_adjust(left=0.12, right=0.93, top=0.92, bottom=0.10)

    # --- save ---
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_format in ("pdf", "both"):
        fig.savefig(str(output_dir / "fig_pose_scatter.pdf"), dpi=300)
    if output_format in ("png", "both"):
        fig.savefig(str(output_dir / "fig_pose_scatter.png"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sanitize filename helper
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    """Replace characters unsafe for directory names with underscores."""
    return re.sub(r"[^\w\-]", "_", name)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_pose_visualization(
    model: torch.nn.Module,
    dataset: MemmapDataset,
    device: torch.device,
    output_dir: Path,
    output_format: str = "both",
    figure_width: float | None = None,
    figure_height: float | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
) -> None:
    """Orchestrate pose joint scatter visualization.

    Parameters
    ----------
    model : nn.Module
        Trained WiFlow model in eval mode.
    dataset : MemmapDataset
        MemmapDataset already filtered to target envs (split="all", envs=...).
    device : torch.device
    output_dir : Path
        Base output directory; ``pose_viz/`` is created underneath.
    output_format : str
        ``"png"``, ``"pdf"``, or ``"both"`` (default).
    figure_width : float | None
    figure_height : float | None
    batch_size : int
    num_workers : int
    """
    viz_dir = output_dir / "pose_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # --- collect middle frames ---
    print("  Collecting middle frames per (action, subject)...")
    frame_positions = _collect_middle_frames(dataset)
    print(f"  Selected {len(frame_positions)} frames for visualization")

    if not frame_positions:
        print("  [WARN] No frames selected — nothing to visualize")
        return

    # --- single-epoch inference pass ---
    subset = Subset(dataset, frame_positions)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=memmap_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    print("  Running inference and saving scatter plots...")
    sample_idx = 0
    with torch.no_grad():
        for batch in loader:
            model_input, target = prepare_model_input(batch, device)
            preds = extract_prediction_keypoints(model(model_input)).cpu().numpy()
            targets_np = target.cpu().numpy()

            for i in range(len(preds)):
                action = str(batch["action"][i])
                subject = str(batch["sample"][i])
                environment = str(batch["environment"][i])
                sample_dir = viz_dir / (
                    f"{_sanitize_name(action)}_"
                    f"{_sanitize_name(environment)}_"
                    f"{_sanitize_name(subject)}"
                )
                _draw_pose_scatter(
                    target=targets_np[i],
                    prediction=preds[i],
                    action=action,
                    subject=subject,
                    environment=environment,
                    output_dir=sample_dir,
                    output_format=output_format,
                    figure_width=figure_width,
                    figure_height=figure_height,
                )
                sample_idx += 1

    print(f"  Saved {sample_idx} scatter plots to {viz_dir}")
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "from evaluation.pose_viz import run_pose_visualization; print('OK')"`
Expected: `OK` (no import errors).

- [ ] **Step 3: Commit**

```bash
git add evaluation/pose_viz.py
git commit -m "feat: add pose_viz module for per-subject joint scatter plots"
```

---

### Task 3: Wire --pose-viz into eval.py

**Files:**
- Modify: `eval.py`

- [ ] **Step 1: Add `--pose-viz` argument to `parse_args`**

Insert after the `--feature-viz` argument block (~line 257–258):
```python
    parser.add_argument(
        "--pose-viz", action="store_true", default=False,
        help="Generate per-subject joint scatter plots (GT vs Prediction).",
    )
```

- [ ] **Step 2: Add mutual exclusion check in `main`**

Insert after `args = parse_args()` and before model loading (~line 279–281):
```python
    if args.feature_viz and args.pose_viz:
        parser.error("--feature-viz and --pose-viz are mutually exclusive")
```

- [ ] **Step 3: Wire `--pose-viz` call after metric evaluation**

Replace the current `--feature-viz` block at the end of `main` (~lines 319–337) with an if/elif:

```python
    # --- pose visualization (optional, separate pass) ---
    if args.pose_viz:
        from evaluation.pose_viz import run_pose_visualization

        print("\n--- Pose Joint Scatter Visualization ---")
        # Use the dataset object (pre-Subset) for metadata queries.
        # If exclude_indices removed rows, we need a fresh dataset for
        # middle-frame queries; reconstruct without exclusion.
        viz_dataset = MemmapDataset(
            data_dir=args.dataset_root,
            split="all",
            envs=eval_envs,
        )
        run_pose_visualization(
            model=model,
            dataset=viz_dataset,
            device=device,
            output_dir=output_dir,
            output_format=args.output_format,
            figure_width=args.figure_width,
            figure_height=args.figure_height,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print("Pose visualization complete.")

    # --- feature visualization (optional, separate pass) ---
    elif args.feature_viz:
        from evaluation.feature_viz import run_feature_visualization

        print("\n--- Feature Visualization ---")
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
        print("Feature visualization complete.")
```

- [ ] **Step 4: Verify eval.py parses --pose-viz correctly**

Run: `python eval.py --help`
Expected: `--pose-viz` appears in help output.

- [ ] **Step 5: Verify mutual exclusion works**

Run: `python eval.py --dataset-root data/mmfi_pose --checkpoint dummy.pth --pose-viz --feature-viz`
Expected: Error message about mutual exclusion.

- [ ] **Step 6: Commit**

```bash
git add eval.py
git commit -m "feat: add --pose-viz flag to eval.py for joint scatter visualization"
```

---

### Task 4: End-to-end manual verification

- [ ] **Step 1: Run pose-viz on a real checkpoint**

```bash
python eval.py \
  --dataset-root data/mmfi_pose \
  --checkpoint runs/source_only_baseline/best_val_pck_0_2.pth \
  --eval-envs env2 \
  --pose-viz \
  --output-dir outputs/pose_viz_test
```
Expected: ~270 directories created under `outputs/pose_viz_test/pose_viz/`, each containing `fig_pose_scatter.png`.

- [ ] **Step 2: Visually inspect 2–3 output PNGs**

Open a few `fig_pose_scatter.png` files and verify:
- White background with subtle grid
- 18 colored joint points (GT dashed rings, Prediction filled circles)
- Error vectors connecting GT → Prediction
- Joint index labels (0–17) near prediction points
- Legend with GT/Prediction entries
- Title includes action / subject / environment

- [ ] **Step 3: Verify feature_viz still works after fig0 removal**

Run: `python eval.py --dataset-root data/mmfi_pose --checkpoint runs/source_only_baseline/best_val_pck_0_2.pth --feature-viz --num-action-samples 1 --output-dir outputs/fv_test`
Expected: Feature viz runs without errors, no fig0_joint_scatter files produced.
