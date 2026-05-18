"""
Visualize a single frame of OpenPose18 ground truth keypoints.

Displays all 18 joints with index labels and (x, y) coordinates.
Draws skeleton edges between connected joints for visual context.

Usage:
    python scripts/visualize_gt.py --gt data/gt_merged/ground_truth.npy --frame 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.skeleton import NUM_OPENPOSE_KEYPOINTS, OPENPOSE_BONE_EDGES

matplotlib.use("TkAgg")

JOINT_NAMES = [
    "Nose(0)",
    "Neck(1)",
    "R_Shoulder(2)",
    "R_Elbow(3)",
    "R_Wrist(4)",
    "L_Shoulder(5)",
    "L_Elbow(6)",
    "L_Wrist(7)",
    "R_Hip(8)",
    "R_Knee(9)",
    "R_Ankle(10)",
    "L_Hip(11)",
    "L_Knee(12)",
    "L_Ankle(13)",
    "R_Eye(14)",
    "L_Eye(15)",
    "R_Ear(16)",
    "L_Ear(17)",
]

JOINT_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1",
]


def main():
    parser = argparse.ArgumentParser(description="Visualize OpenPose18 GT keypoints")
    parser.add_argument("--gt", default="data/gt_merged/ground_truth.npy")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", default=None, help="Save to file instead of showing")
    parser.add_argument("--no-bones", action="store_true", default=False, help="Hide bone connections")
    args = parser.parse_args()

    gt_path = Path(args.gt)
    if not gt_path.is_file():
        print(f"ERROR: ground_truth.npy not found at {gt_path}")
        sys.exit(1)

    gt = np.load(str(gt_path))
    print(f"Loaded ground_truth: {gt.shape} {gt.dtype}")

    if args.frame >= gt.shape[0]:
        print(f"ERROR: frame {args.frame} out of range (0–{gt.shape[0] - 1})")
        sys.exit(1)

    kpts = gt[args.frame]

    valid_mask = ~(np.all(np.isclose(kpts, 0.0), axis=-1))
    n_valid = int(valid_mask.sum())
    print(f"Frame {args.frame}: {n_valid}/{NUM_OPENPOSE_KEYPOINTS} joints valid")

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_facecolor("#0a0a1a")

    for i in range(NUM_OPENPOSE_KEYPOINTS):
        x, y = kpts[i]
        color = JOINT_COLORS[i]
        ax.scatter(x, y, c=color, s=120, edgecolors="white", linewidths=1.0, zorder=5)
        ax.annotate(
            f"{JOINT_NAMES[i]}\n({x:.4f}, {y:.4f})",
            (x, y),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=7,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a1a", edgecolor=color, alpha=0.85),
            zorder=10,
        )

    if not args.no_bones:
        for start, end in OPENPOSE_BONE_EDGES:
            if valid_mask[start] and valid_mask[end]:
                ax.plot(
                    [kpts[start, 0], kpts[end, 0]],
                    [kpts[start, 1], kpts[end, 1]],
                    color="#ffffff",
                    linewidth=1.0,
                    alpha=0.35,
                    zorder=1,
                )

    margin = 0.15
    valid_kpts = kpts[valid_mask]
    if len(valid_kpts) > 0:
        x_min, x_max = valid_kpts[:, 0].min(), valid_kpts[:, 0].max()
        y_min, y_max = valid_kpts[:, 1].min(), valid_kpts[:, 1].max()
        x_range = max(x_max - x_min, 0.5)
        y_range = max(y_max - y_min, 0.5)
        ax.set_xlim(x_min - x_range * 0.3, x_max + x_range * 0.3)
        ax.set_ylim(y_max + y_range * 0.3, y_min - y_range * 0.3)
    else:
        ax.set_xlim(-margin, margin)
        ax.set_ylim(margin, -margin)

    ax.set_aspect("equal")
    ax.set_title(
        f"OpenPose18 Ground Truth — Frame {args.frame} ({n_valid}/18 joints)",
        fontsize=14,
        fontweight="bold",
        color="white",
        pad=15,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    fig.patch.set_facecolor("#0a0a1a")
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()