"""Pose joint scatter + skeleton visualization for eval.py --pose-viz.

Generates per-action figures: individual two-subplot (scatter + skeleton)
files and a per-action NxM composite grid of skeleton overlays.
"""

from __future__ import annotations

from collections import defaultdict
from math import ceil
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

BONE_EDGES: list[tuple[int, int]] = [
    (4, 7), (7, 3),
    (3, 9), (3, 6), (3, 11),
    (9, 13), (13, 10), (11, 8), (8, 12),
    (6, 0),
    (0, 15), (0, 16),
    (15, 14), (14, 17), (16, 5), (5, 1), (1, 2),
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
# Axis-level drawing
# ---------------------------------------------------------------------------

def _compute_axes_limits(
    target: np.ndarray,
    prediction: np.ndarray,
    pad_ratio: float = 0.1,
    min_pad: float = 0.02,
) -> tuple[float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max) with padding."""
    all_points = np.concatenate([target, prediction], axis=0)
    x_min, x_max = float(all_points[:, 0].min()), float(all_points[:, 0].max())
    y_min, y_max = float(all_points[:, 1].min()), float(all_points[:, 1].max())
    x_pad = max((x_max - x_min) * pad_ratio, min_pad)
    y_pad = max((y_max - y_min) * pad_ratio, min_pad)
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def _draw_scatter(ax: plt.Axes, target: np.ndarray, prediction: np.ndarray) -> None:
    """Draw GT (hollow) and prediction (filled) joint scatter on *ax*."""
    # error vectors
    for j in range(18):
        ax.plot(
            [target[j, 0], prediction[j, 0]],
            [target[j, 1], prediction[j, 1]],
            color="#cccccc", linewidth=0.5, linestyle="--", zorder=1,
        )

    # GT: dashed hollow circles
    for j in range(18):
        ax.scatter(
            target[j, 0], target[j, 1],
            facecolors="none",
            edgecolors=JOINT_COLORS[j],
            marker="o", s=80, linewidths=1.8,
            zorder=3,
        )

    # Prediction: filled circles with dark border
    for j in range(18):
        ax.scatter(
            prediction[j, 0], prediction[j, 1],
            facecolors=JOINT_COLORS[j],
            edgecolors="#333333",
            marker="o", s=80, linewidths=1.0, zorder=2,
        )
        ax.annotate(
            str(j),
            (prediction[j, 0], prediction[j, 1]),
            fontsize=8, fontweight="bold",
            xytext=(8, -8), textcoords="offset points",
        )

    # legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="#888888", markersize=8, markeredgewidth=1.8,
               linestyle="--", label="GT"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markeredgecolor="#333333", markersize=8, label="Prediction"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)


def _draw_skeleton(
    ax: plt.Axes,
    keypoints: np.ndarray,
    *,
    hollow: bool = False,
    bone_linestyle: str = "-",
    bone_color: str = "#333333",
    bone_linewidth: float = 1.2,
    marker_size: float = 60,
    base_zorder: int = 1,
) -> None:
    """Draw skeleton bones and joint circles for one set of keypoints.

    Parameters
    ----------
    hollow : bool
        If True, joint circles are hollow (GT style); otherwise filled (Pred style).
    """
    # bones
    for i, j in BONE_EDGES:
        ax.plot(
            [keypoints[i, 0], keypoints[j, 0]],
            [keypoints[i, 1], keypoints[j, 1]],
            color=bone_color, linestyle=bone_linestyle,
            linewidth=bone_linewidth, alpha=0.7, zorder=base_zorder,
        )

    # joints
    for j in range(18):
        if hollow:
            ax.scatter(
                keypoints[j, 0], keypoints[j, 1],
                facecolors="none",
                edgecolors=JOINT_COLORS[j],
                marker="o", s=marker_size, linewidths=1.5,
                zorder=base_zorder + 1,
            )
        else:
            ax.scatter(
                keypoints[j, 0], keypoints[j, 1],
                facecolors=JOINT_COLORS[j],
                edgecolors="#333333",
                marker="o", s=marker_size, linewidths=1.0,
                zorder=base_zorder + 1,
            )


# ---------------------------------------------------------------------------
# Figure-level
# ---------------------------------------------------------------------------

def _save_individual(
    target: np.ndarray,
    prediction: np.ndarray,
    action: str,
    subject: str,
    environment: str,
    output_dir: Path,
    figure_width: float | None,
    figure_height: float | None,
) -> None:
    """Save a two-subplot figure: scatter (left) + skeleton (right)."""
    fig_w = figure_width or 14.0
    fig_h = figure_height or 6.5
    fig, (ax_scatter, ax_skeleton) = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    # --- common axis limits ---
    x_min, x_max, y_min, y_max = _compute_axes_limits(target, prediction)

    # --- left: scatter ---
    ax_scatter.set_facecolor("#ffffff")
    ax_scatter.grid(True, alpha=0.3, color="#e8e8e8", linewidth=0.5)
    _draw_scatter(ax_scatter, target, prediction)
    ax_scatter.set_xlim(x_min, x_max)
    ax_scatter.set_ylim(y_max, y_min)  # invert Y for natural pose
    ax_scatter.set_aspect("equal")
    ax_scatter.set_xlabel("Normalized X")
    ax_scatter.set_ylabel("Normalized Y")
    ax_scatter.set_title("Joint Scatter (GT vs Pred)", fontsize=11, fontweight="bold")

    # --- right: skeleton ---
    ax_skeleton.set_facecolor("#fafafa")
    ax_skeleton.grid(True, alpha=0.2, color="#d0d0d0", linewidth=0.5)
    _draw_skeleton(ax_skeleton, target,
                   hollow=True, bone_linestyle="--", bone_color="#aaaaaa",
                   base_zorder=1)
    _draw_skeleton(ax_skeleton, prediction,
                   hollow=False, bone_linestyle="-", bone_color="#333333",
                   base_zorder=3)
    ax_skeleton.set_xlim(x_min, x_max)
    ax_skeleton.set_ylim(y_max, y_min)
    ax_skeleton.set_aspect("equal")
    ax_skeleton.set_xlabel("Normalized X")
    ax_skeleton.set_ylabel("Normalized Y")
    ax_skeleton.set_title("Skeleton (GT vs Pred)", fontsize=11, fontweight="bold")

    # skeleton legend
    legend_elements = [
        Line2D([0], [0], color="#aaaaaa", linestyle="--", linewidth=1.2, label="GT"),
        Line2D([0], [0], color="#333333", linestyle="-", linewidth=1.2, label="Prediction"),
    ]
    ax_skeleton.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.suptitle(
        f"{action} / {subject} / {environment}",
        fontsize=13, fontweight="bold",
    )
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.10, wspace=0.25)

    # save
    action_dir = output_dir / action
    action_dir.mkdir(parents=True, exist_ok=True)
    safe_subject = subject.replace("/", "_").replace("\\", "_")
    safe_env = environment.replace("/", "_").replace("\\", "_")
    fig.savefig(str(action_dir / f"{safe_subject}_{safe_env}.png"), dpi=300)
    plt.close(fig)


def _build_action_composite(
    action: str,
    samples: list[dict],
    output_dir: Path,
) -> None:
    """Build an N×M grid of skeleton overlays for all samples of *action*."""
    n = len(samples)
    if n == 0:
        return

    rows = int(ceil(n ** 0.5))
    cols = int(ceil(n / rows))
    cell_w = 4.5
    cell_h = 4.5

    fig, axes = plt.subplots(rows, cols, figsize=(cell_w * cols, cell_h * rows + 0.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # global limits from all samples
    all_points = np.concatenate(
        [np.concatenate([s["target"], s["prediction"]], axis=0) for s in samples],
        axis=0,
    )
    x_min, x_max = float(all_points[:, 0].min()), float(all_points[:, 0].max())
    y_min, y_max = float(all_points[:, 1].min()), float(all_points[:, 1].max())
    x_pad = max((x_max - x_min) * 0.1, 0.02)
    y_pad = max((y_max - y_min) * 0.1, 0.02)
    gx_min, gx_max = x_min - x_pad, x_max + x_pad
    gy_min, gy_max = y_min - y_pad, y_max + y_pad

    for idx, sample in enumerate(samples):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.set_facecolor("#fafafa")
        ax.grid(True, alpha=0.15, color="#d0d0d0", linewidth=0.3)

        _draw_skeleton(ax, sample["target"],
                       hollow=True, bone_linestyle="--", bone_color="#aaaaaa",
                       bone_linewidth=0.8, marker_size=25, base_zorder=1)
        _draw_skeleton(ax, sample["prediction"],
                       hollow=False, bone_linestyle="-", bone_color="#333333",
                       bone_linewidth=0.8, marker_size=25, base_zorder=3)

        ax.set_xlim(gx_min, gx_max)
        ax.set_ylim(gy_max, gy_min)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{sample['subject']} / {sample['environment']}",
                     fontsize=8, fontweight="bold")

    # hide unused cells
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"Action: {action}  ({n} subjects)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    action_dir = output_dir / action
    action_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(action_dir / "_composite.png"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_pose_visualization(
    model: torch.nn.Module,
    dataset: MemmapDataset,
    device: torch.device,
    output_dir: Path,
    figure_width: float | None = None,
    figure_height: float | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
) -> None:
    """Orchestrate pose joint scatter + skeleton visualization.

    Parameters
    ----------
    model : nn.Module
        Trained WiFlow model in eval mode.
    dataset : MemmapDataset
        MemmapDataset already filtered to target envs (split="all", envs=...).
    device : torch.device
    output_dir : Path
        Base output directory; ``pose_viz/`` is created underneath.
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

    print("  Running inference and saving figures...")
    action_samples: dict[str, list[dict]] = defaultdict(list)
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

                _save_individual(
                    target=targets_np[i],
                    prediction=preds[i],
                    action=action,
                    subject=subject,
                    environment=environment,
                    output_dir=viz_dir,
                    figure_width=figure_width,
                    figure_height=figure_height,
                )
                action_samples[action].append({
                    "target": targets_np[i],
                    "prediction": preds[i],
                    "subject": subject,
                    "environment": environment,
                })
                sample_idx += 1

    # --- per-action composites ---
    print(f"  Building per-action composite figures ({len(action_samples)} actions)...")
    for action in sorted(action_samples):
        _build_action_composite(action, action_samples[action], viz_dir)

    print(f"  Saved {sample_idx} individual figures + "
          f"{len(action_samples)} composites to {viz_dir}")
