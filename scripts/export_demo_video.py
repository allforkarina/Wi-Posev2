#!/usr/bin/env python3
"""Export a side-by-side GT-vs-Pred pose comparison video (MP4 + GIF).

Generates a frame-by-frame video from a trained WiFlow checkpoint,
showing ground-truth skeleton (left) vs model prediction (right)
across a full action sequence.

Usage:
    python scripts/export_demo_video.py \\
        --checkpoint outputs/train/best_val_pck_0_2.pth \\
        --dataset-root data/mmfi_pose \\
        --action 0
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project path setup (no package install — scripts add root to sys.path)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.memmap_dataset import MemmapDataset
from data.split_manifest import load_manifest
from eval import build_evaluation_dataset, load_checkpoint_model
from evaluation.pose_viz import (
    BONE_EDGES,
    JOINT_COLORS,
    _compute_axes_limits,
    _draw_skeleton,
)
from train import extract_prediction_keypoints, prepare_model_input

# Reset rcParams that pose_viz overrides (we need exact pixel dimensions for video)
matplotlib.rcParams["savefig.bbox"] = None
# Keep this overridden — pose_viz needs it
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ---------------------------------------------------------------------------
# Resolution presets
# ---------------------------------------------------------------------------

RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}


def parse_resolution(raw: str) -> tuple[int, int]:
    """Parse a resolution string like '1280x720' or preset name like '720p'."""
    if raw in RESOLUTION_PRESETS:
        return RESOLUTION_PRESETS[raw]
    parts = raw.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Resolution must be WxH or a preset {list(RESOLUTION_PRESETS)}, got {raw!r}"
        )
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid resolution: {raw!r}")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("Resolution dimensions must be positive")
    if w % 2 or h % 2:
        raise argparse.ArgumentTypeError(
            "Resolution dimensions must be even for H.264 yuv420p output"
        )
    return w, h


def positive_float(raw: str) -> float:
    """Parse a strictly positive floating-point CLI value."""
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a number, got {raw!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero")
    return value


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_mpjpe(gt: np.ndarray, pred: np.ndarray) -> float:
    """Mean Per Joint Position Error (Euclidean distance, averaged over 18 joints).

    Parameters
    ----------
    gt : np.ndarray, shape [18, 2]
        Ground-truth keypoints in normalized coordinates.
    pred : np.ndarray, shape [18, 2]
        Predicted keypoints in normalized coordinates.

    Returns
    -------
    float
        Mean Euclidean distance across all 18 joints.
    """
    if gt.shape != (18, 2) or pred.shape != (18, 2):
        raise ValueError(f"Expected [18, 2] keypoints, got gt={gt.shape} pred={pred.shape}")
    per_joint = np.linalg.norm(pred - gt, axis=1)  # [18]
    return float(per_joint.mean())


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_action_frames(
    dataset: MemmapDataset,
    action: str,
    subject: str | None = None,
) -> tuple[list[dict], list[str]]:
    """Collect ordered frames and available subjects for an exact action label.

    Parameters
    ----------
    dataset : MemmapDataset
        Dataset in 'all' split mode.
    action : str
        Exact action label stored in the dataset metadata.
    subject : str | None
        Exact subject label. Selects the first sorted subject when omitted.

    Returns
    -------
    tuple[list[dict], list[str]]
        Ordered frame dictionaries and the available subject labels.
    """
    available_actions = sorted({
        str(dataset._actions[int(index)]) for index in dataset.indices
    })
    if action not in available_actions:
        raise ValueError(
            f"No frames found for action={action!r}. "
            f"Available actions: {', '.join(available_actions)}"
        )

    matching = [
        position
        for position, index in enumerate(dataset.indices)
        if str(dataset._actions[int(index)]) == action
    ]
    samples = [
        str(dataset._samples[int(dataset.indices[position])])
        for position in matching
    ]
    available_subjects = sorted(set(samples))
    selected_subject = subject if subject is not None else available_subjects[0]
    if selected_subject not in available_subjects:
        raise ValueError(
            f"No frames found for action={action!r}, subject={selected_subject!r}. "
            f"Available subjects: {', '.join(available_subjects)}"
        )

    frames = [
        dataset[position]
        for position, sample in zip(matching, samples)
        if sample == selected_subject
    ]
    frames.sort(key=lambda frame: frame["meta"]["frame_idx"])
    return frames, available_subjects


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

# Style constants for the demo video
GT_COLOR = "#4a9eff"       # blue
PRED_COLOR = "#ff6b6b"     # red
GT_BONE_STYLE = "--"
PRED_BONE_STYLE = "-"
BG_COLOR = "#0d0d1a"       # dark background
PANEL_BG = "#111122"
TEXT_COLOR = "#cccccc"
ACCENT_COLOR = "#fdcb6e"   # yellow for labels
PROGRESS_FILL = "#4a9eff"
PROGRESS_BG = "#333333"


def render_frame(
    gt: np.ndarray,
    pred: np.ndarray,
    mpjpe: float,
    frame_idx: int,
    total_frames: int,
    action_label: str,
    subject_id: str,
    model_label: str,
    output_dir: Path,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Render a single side-by-side comparison frame and save as PNG.

    Layout (1280×720 at 100 DPI → 12.8×7.2 inches):
        ┌─────────────────────────────────────────────┐
        │  Action: X  Subject: Y        MPJPE: Z mm   │  ← suptitle row
        ├──────────────────────┬──┬───────────────────┤
        │    GT Skeleton       │  │   Pred Skeleton   │
        │    (blue dashed      │  │   (red solid      │
        │     + hollow ○)      │  │    + filled ●)    │
        ├──────────────────────┴──┴───────────────────┤
        │  ████████░░░░░░░░  Frame N / Total          │  ← progress bar
        └─────────────────────────────────────────────┘

    Parameters
    ----------
    gt, pred : np.ndarray, shape [18, 2]
    mpjpe : float
        Mean per-joint position error.
    frame_idx : int
        0-based frame index.
    total_frames : int
        Total number of frames in the sequence.
    action_label, subject_id : str
        Metadata for the overlay labels.
    model_label : str
        Label for the prediction panel (e.g., "WiFlow v2").
    output_dir : Path
        Directory to save the PNG into.
    width, height : int
        Output pixel dimensions (default 1280×720).

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    if gt.shape != (18, 2) or pred.shape != (18, 2):
        raise ValueError(f"Expected [18, 2] keypoints, got gt={gt.shape} pred={pred.shape}")

    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=BG_COLOR)

    # --- compute shared axis limits from both GT and pred ---
    x_min, x_max, y_min, y_max = _compute_axes_limits(gt, pred)

    # --- GridSpec layout ---
    # Top info bar (8% height), main panels (84%), bottom progress bar (8%)
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[0.08, 0.84, 0.08],
        width_ratios=[1, 1],
        hspace=0.05, wspace=0.02,
        left=0.02, right=0.98, top=0.96, bottom=0.02,
    )

    # --- Top info bar (spans both columns) ---
    ax_info = fig.add_subplot(gs[0, :])
    ax_info.set_facecolor(BG_COLOR)
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for spine in ax_info.spines.values():
        spine.set_visible(False)

    # Left: action + subject
    ax_info.text(
        0.01, 0.5,
        f"Action: {action_label}    Subject: {subject_id}",
        transform=ax_info.transAxes,
        color=ACCENT_COLOR, fontsize=11, fontweight="bold",
        va="center", ha="left",
    )
    # Right: MPJPE
    ax_info.text(
        0.99, 0.5,
        f"MPJPE: {mpjpe:.2f}",
        transform=ax_info.transAxes,
        color="#00cec9", fontsize=11, fontweight="bold",
        va="center", ha="right",
    )

    # --- Left panel: GT ---
    ax_gt = fig.add_subplot(gs[1, 0])
    ax_gt.set_facecolor(PANEL_BG)
    ax_gt.set_xlim(x_min, x_max)
    ax_gt.set_ylim(y_max, y_min)  # invert Y for natural pose orientation
    ax_gt.set_aspect("equal")
    ax_gt.set_xticks([])
    ax_gt.set_yticks([])
    for spine in ax_gt.spines.values():
        spine.set_color("#333355")
        spine.set_linewidth(1.5)

    _draw_skeleton(
        ax_gt, gt,
        hollow=True,
        bone_linestyle=GT_BONE_STYLE,
        bone_color=GT_COLOR,
        bone_linewidth=1.5,
        marker_size=60,
        base_zorder=1,
    )
    ax_gt.set_title("Ground Truth", color=GT_COLOR, fontsize=13, fontweight="bold", pad=8)

    # --- Right panel: Pred ---
    ax_pred = fig.add_subplot(gs[1, 1])
    ax_pred.set_facecolor(PANEL_BG)
    ax_pred.set_xlim(x_min, x_max)
    ax_pred.set_ylim(y_max, y_min)
    ax_pred.set_aspect("equal")
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])
    for spine in ax_pred.spines.values():
        spine.set_color("#553333")
        spine.set_linewidth(1.5)

    _draw_skeleton(
        ax_pred, pred,
        hollow=False,
        bone_linestyle=PRED_BONE_STYLE,
        bone_color=PRED_COLOR,
        bone_linewidth=1.5,
        marker_size=60,
        base_zorder=1,
    )
    ax_pred.set_title(model_label, color=PRED_COLOR, fontsize=13, fontweight="bold", pad=8)

    # --- Bottom progress bar (spans both columns) ---
    ax_bar = fig.add_subplot(gs[2, :])
    ax_bar.set_facecolor(BG_COLOR)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    # Progress bar background
    bar_left = 0.02
    bar_width = 0.70
    bar_bottom = 0.25
    bar_height = 0.50

    ax_bar.add_patch(
        plt.Rectangle(
            (bar_left, bar_bottom), bar_width, bar_height,
            facecolor=PROGRESS_BG, edgecolor="#555555",
            linewidth=0.5, transform=ax_bar.transAxes,
            zorder=1,
        )
    )
    # Progress bar fill
    progress = (frame_idx + 1) / total_frames
    ax_bar.add_patch(
        plt.Rectangle(
            (bar_left, bar_bottom), bar_width * progress, bar_height,
            facecolor=PROGRESS_FILL, edgecolor="none",
            transform=ax_bar.transAxes, zorder=2,
        )
    )
    # Frame counter
    ax_bar.text(
        bar_left + bar_width + 0.02, 0.5,
        f"Frame {frame_idx + 1} / {total_frames}",
        transform=ax_bar.transAxes,
        color=TEXT_COLOR, fontsize=10,
        va="center", ha="left",
    )

    # --- Save ---
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"frame_{frame_idx:03d}.png"
    fig.savefig(str(output_path), dpi=dpi, facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# FFmpeg check
# ---------------------------------------------------------------------------


def check_ffmpeg() -> None:
    """Check that ffmpeg is available on PATH. Exit with instructions if not."""
    if shutil.which("ffmpeg") is None:
        print(
            "Error: ffmpeg not found on PATH.\n"
            "Install it with:\n"
            "  conda install -c conda-forge ffmpeg\n"
            "or:\n"
            "  sudo apt install ffmpeg",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Video encoding
# ---------------------------------------------------------------------------


def encode_mp4(
    frames_dir: Path,
    output_path: Path,
    fps: float = 18,
    width: int = 1280,
    height: int = 720,
) -> None:
    """Encode a sequence of PNG frames to H.264 MP4 via ffmpeg.

    Parameters
    ----------
    frames_dir : Path
        Directory containing frame_000.png, frame_001.png, ...
    output_path : Path
        Destination .mp4 file.
    fps : float
        Frames per second.
    width, height : int
        Output dimensions (ffmpeg will scale if input differs).
    """
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%03d.png"),
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x0d0d1a",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg MP4 error:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"ffmpeg MP4 encoding failed with code {result.returncode}")


def encode_gif(
    frames_dir: Path,
    output_path: Path,
    fps: float = 18,
    width: int = 1280,
    height: int = 720,
) -> None:
    """Encode a sequence of PNG frames to an animated GIF via ffmpeg.

    Uses the two-pass palettegen + paletteuse approach for quality.

    Parameters
    ----------
    frames_dir : Path
        Directory containing frame_000.png, frame_001.png, ...
    output_path : Path
        Destination .gif file.
    fps : float
        Frames per second.
    width, height : int
        Output dimensions.
    """
    palette_path = frames_dir / "_palette.png"

    # Pass 1: generate palette
    cmd_palette = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%03d.png"),
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x0d0d1a,"
               "palettegen=stats_mode=diff",
        str(palette_path),
    ]
    result = subprocess.run(cmd_palette, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg palette error:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"ffmpeg palette generation failed with code {result.returncode}")

    # Pass 2: encode GIF using palette
    cmd_gif = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%03d.png"),
        "-i", str(palette_path),
        "-lavfi", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                  f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x0d0d1a [x]; "
                  "[x][1:v] paletteuse=dither=bayer:bayer_scale=5",
        str(output_path),
    ]
    result = subprocess.run(cmd_gif, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg GIF error:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"ffmpeg GIF encoding failed with code {result.returncode}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_export_model_and_dataset(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, MemmapDataset]:
    """Load a checkpoint and its manifest-consistent evaluation dataset."""
    manifest = (
        load_manifest(args.split_manifest, args.dataset_root)
        if args.split_manifest
        else None
    )
    model = load_checkpoint_model(
        args.checkpoint,
        device,
        expected_manifest_hash=manifest.manifest_hash if manifest else None,
    )
    dataset = build_evaluation_dataset(
        dataset_root=args.dataset_root,
        manifest=manifest,
        manifest_key=args.manifest_key,
    )
    return model, dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export side-by-side GT-vs-Pred pose comparison video.",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a WiFlow checkpoint (.pth).",
    )
    parser.add_argument(
        "--dataset-root", required=True,
        help="Path to the NPY memmap dataset directory.",
    )
    parser.add_argument(
        "--action", required=True,
        help="Exact action label to visualize (e.g. A01).",
    )
    parser.add_argument(
        "--subject", default=None,
        help="Exact subject label (e.g. S03). Uses the first available if omitted.",
    )
    parser.add_argument(
        "--output-dir", default="outputs/demo_videos",
        help="Output directory for video files.",
    )
    parser.add_argument(
        "--fps", type=positive_float, default=18.0,
        help="Frames per second (default: 18).",
    )
    parser.add_argument(
        "--resolution", type=parse_resolution, default=RESOLUTION_PRESETS["720p"],
        help="Output resolution: WxH or preset (720p, 1080p). Default: 1280x720.",
    )
    parser.add_argument(
        "--model-label", default="WiFlow v2",
        help="Label shown on the prediction panel.",
    )
    parser.add_argument(
        "--keep-frames", action="store_true", default=False,
        help="Keep intermediate PNG frames after encoding.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--gif-only", action="store_true", default=False,
        help="Generate only GIF (skip MP4).",
    )
    output_group.add_argument(
        "--mp4-only", action="store_true", default=False,
        help="Generate only MP4 (skip GIF).",
    )
    parser.add_argument(
        "--split-manifest", default=None,
        help="Path to a deterministic split manifest.",
    )
    parser.add_argument(
        "--manifest-key", default=None,
        help="Named array in --split-manifest (e.g. env2_test).",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for inference (default: cuda).",
    )
    args = parser.parse_args()
    if bool(args.split_manifest) != bool(args.manifest_key):
        parser.error("--split-manifest and --manifest-key must be provided together")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    check_ffmpeg()

    width, height = args.resolution
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Resolution: {width}×{height}")
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")

    # --- 1. Load model and dataset ---
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Loading dataset: {args.dataset_root}")
    model, dataset = load_export_model_and_dataset(args, device)

    # --- 2. Collect frames ---
    print(f"Collecting frames for action={args.action}, subject={args.subject}...")
    frames, available_subjects = collect_action_frames(
        dataset, action=args.action, subject=args.subject
    )
    print(f"  Found {len(frames)} frames")

    actual_subject_id = frames[0]["meta"]["subject"]
    actual_action = frames[0]["meta"]["action"]
    print(f"  Available subjects: {', '.join(available_subjects)}")
    print(f"  Selected subject: {actual_subject_id}")

    # --- 3. Run inference ---
    print("Running inference...")
    results: list[dict] = []
    with torch.no_grad():
        for i, frame in enumerate(frames):
            # Build a batch-of-1 dict matching memmap_collate_fn output
            csi = frame["csi"].unsqueeze(0)  # [1, H, W, C]
            csi = csi.permute(0, 2, 3, 1).contiguous()  # [1, C, H, W]
            kpts = frame["kpts18"].unsqueeze(0)  # [1, 18, 2]

            batch = {
                "csi_amplitude": csi,
                "keypoints": kpts,
            }
            model_input, target = prepare_model_input(batch, device)
            pred = extract_prediction_keypoints(model(model_input))

            gt_np = target.squeeze(0).cpu().numpy()  # [18, 2]
            pred_np = pred.squeeze(0).cpu().numpy()  # [18, 2]
            mpjpe = compute_mpjpe(gt_np, pred_np)

            results.append({
                "gt": gt_np,
                "pred": pred_np,
                "mpjpe": mpjpe,
                "frame_idx": i,
            })

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Frame {i + 1}/{len(frames)} — MPJPE: {mpjpe:.4f}")

    print(f"  Inference complete. {len(results)} frames processed.")

    # --- 4. Render frames ---
    print("Rendering frames...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        frames_dir = Path(tmp_dir)
        for i, r in enumerate(results):
            render_frame(
                gt=r["gt"],
                pred=r["pred"],
                mpjpe=r["mpjpe"],
                frame_idx=i,
                total_frames=len(results),
                action_label=actual_action,
                subject_id=actual_subject_id,
                model_label=args.model_label,
                output_dir=frames_dir,
                width=width,
                height=height,
            )
            if (i + 1) % 50 == 0:
                print(f"  Rendered {i + 1}/{len(results)} frames")

        # --- 5. Encode video ---
        safe_action = actual_action.replace("/", "_").replace("\\", "_")
        safe_subject = actual_subject_id.replace("/", "_").replace("\\", "_")
        base_name = f"action_{safe_action}_subject_{safe_subject}_{width}x{height}_{args.fps:.0f}fps"

        if not args.gif_only:
            mp4_path = output_dir / f"{base_name}.mp4"
            print(f"Encoding MP4: {mp4_path}")
            encode_mp4(frames_dir, mp4_path, fps=args.fps, width=width, height=height)
            print(f"  MP4 saved: {mp4_path}")

        if not args.mp4_only:
            gif_path = output_dir / f"{base_name}.gif"
            print(f"Encoding GIF: {gif_path}")
            encode_gif(frames_dir, gif_path, fps=args.fps, width=width, height=height)
            print(f"  GIF saved: {gif_path}")

        # --- 6. Optionally keep frames ---
        if args.keep_frames:
            keep_dir = output_dir / f"frames_{safe_action}_{safe_subject}"
            keep_dir.mkdir(parents=True, exist_ok=True)
            for png in sorted(frames_dir.glob("frame_*.png")):
                shutil.copy2(png, keep_dir / png.name)
            print(f"  Frames kept at: {keep_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
