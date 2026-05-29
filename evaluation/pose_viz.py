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
