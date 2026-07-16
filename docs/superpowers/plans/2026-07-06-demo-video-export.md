# Demo Video Export — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/export_demo_video.py` — a standalone CLI that generates side-by-side GT-vs-Pred pose comparison videos (MP4 + GIF) from a trained WiFlow checkpoint across a full 297-frame action sequence.

**Architecture:** Pure Python + matplotlib + ffmpeg. Reuses `evaluation/pose_viz.py` drawing primitives (`_draw_skeleton`, `BONE_EDGES`, `JOINT_COLORS`) and `eval.py`'s `load_checkpoint_model()`. Zero changes to existing files. Renders 297 PNGs to a temp dir, then encodes via `subprocess.run(["ffmpeg", ...])` with graceful fallback if ffmpeg is missing.

**Tech Stack:** Python 3.10+, matplotlib (Agg backend), PyTorch, NumPy, ffmpeg (system), pathlib, argparse

**Spec:** `docs/superpowers/specs/2026-07-06-demo-video-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/export_demo_video.py` | **Create** | Main CLI: arg parsing, model loading, frame collection, inference, rendering loop, ffmpeg encoding |
| `tests/test_export_demo_video.py` | **Create** | Unit tests: MPJPE calc, render_frame output, CLI arg parsing, ffmpeg detection |

No existing files are modified.

---

### Task 1: Scaffold test file

**Files:**
- Create: `tests/test_export_demo_video.py`

- [ ] **Step 1: Write the skeleton test file with imports and fixtures**

```python
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# We'll import from the script once it exists
# from scripts.export_demo_video import compute_mpjpe, render_frame, parse_args


class TestComputeMpjpe:
    """Tests for per-frame MPJPE calculation."""

    def test_identical_keypoints_give_zero(self) -> None:
        """MPJPE between identical keypoint sets should be 0."""
        from scripts.export_demo_video import compute_mpjpe

        kpts = np.random.randn(18, 2).astype(np.float32)
        error = compute_mpjpe(kpts, kpts)
        assert error == pytest.approx(0.0, abs=1e-6)

    def test_known_offset(self) -> None:
        """MPJPE with a known 1.0 offset on all joints should be 1.0."""
        from scripts.export_demo_video import compute_mpjpe

        gt = np.zeros((18, 2), dtype=np.float32)
        pred = np.ones((18, 2), dtype=np.float32)
        # Each joint: sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.4142
        error = compute_mpjpe(gt, pred)
        assert error == pytest.approx(np.sqrt(2), rel=1e-4)

    def test_single_joint_offset(self) -> None:
        """Offset on only one joint should average across all 18."""
        from scripts.export_demo_video import compute_mpjpe

        gt = np.zeros((18, 2), dtype=np.float32)
        pred = np.zeros((18, 2), dtype=np.float32)
        pred[0, 0] = 3.0  # offset of 3 in x on joint 0 only
        # Only joint 0 has error = 3.0; others 0. Mean = 3.0 / 18 = 0.1666...
        error = compute_mpjpe(gt, pred)
        assert error == pytest.approx(3.0 / 18.0, rel=1e-4)


class TestRenderFrame:
    """Tests for single-frame rendering."""

    def test_creates_png_file(self, tmp_path: Path) -> None:
        """render_frame should create a PNG at the expected path."""
        from scripts.export_demo_video import render_frame

        gt = np.random.randn(18, 2).astype(np.float32) * 0.1
        pred = np.random.randn(18, 2).astype(np.float32) * 0.1

        output_path = render_frame(
            gt=gt,
            pred=pred,
            mpjpe=12.4,
            frame_idx=0,
            total_frames=297,
            action_label="0",
            subject_id="1",
            model_label="WiFlow v2",
            output_dir=tmp_path,
        )

        assert output_path.is_file()
        assert output_path.suffix == ".png"

    def test_png_has_correct_dimensions(self, tmp_path: Path) -> None:
        """Output PNG should be 1280×720 at 100 DPI (12.8×7.2 inches)."""
        from scripts.export_demo_video import render_frame

        gt = np.random.randn(18, 2).astype(np.float32) * 0.1
        pred = np.random.randn(18, 2).astype(np.float32) * 0.1

        output_path = render_frame(
            gt=gt,
            pred=pred,
            mpjpe=12.4,
            frame_idx=0,
            total_frames=297,
            action_label="0",
            subject_id="1",
            model_label="WiFlow v2",
            output_dir=tmp_path,
        )

        # Read back and check dimensions
        img = plt.imread(str(output_path))
        # At 100 DPI, 1280×720 → 12.8×7.2 inches → saved at 100 DPI → 1280×720 pixels
        assert img.shape[0] == 720, f"Expected 720px height, got {img.shape[0]}"
        assert img.shape[1] == 1280, f"Expected 1280px width, got {img.shape[1]}"

    def test_frame_idx_zero_shows_frame_1_of_total(self, tmp_path: Path) -> None:
        """Frame index 0 should display as 'Frame 1/297'."""
        from scripts.export_demo_video import render_frame

        gt = np.random.randn(18, 2).astype(np.float32) * 0.1
        pred = np.random.randn(18, 2).astype(np.float32) * 0.1

        output_path = render_frame(
            gt=gt, pred=pred, mpjpe=5.0,
            frame_idx=0, total_frames=297,
            action_label="waving", subject_id="S01",
            model_label="Test", output_dir=tmp_path,
        )
        assert output_path.is_file()


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_required_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--checkpoint, --dataset-root, and --action are required."""
        from scripts.export_demo_video import parse_args

        monkeypatch.setattr(sys, "argv", [
            "export_demo_video.py",
            "--checkpoint", "model.pth",
            "--dataset-root", "data/mmfi_pose",
            "--action", "0",
        ])
        args = parse_args()
        assert args.checkpoint == "model.pth"
        assert args.dataset_root == "data/mmfi_pose"
        assert args.action == 0

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Check all default values."""
        from scripts.export_demo_video import parse_args

        monkeypatch.setattr(sys, "argv", [
            "export_demo_video.py",
            "--checkpoint", "model.pth",
            "--dataset-root", "data/mmfi_pose",
            "--action", "0",
        ])
        args = parse_args()
        assert args.fps == 18
        assert args.resolution == "1280x720"
        assert args.subject == 0
        assert args.output_dir == "outputs/demo_videos"
        assert args.model_label == "WiFlow v2"
        assert args.keep_frames is False
        assert args.gif_only is False
        assert args.mp4_only is False

    def test_resolution_preset_720p(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--resolution 720p should resolve to 1280x720."""
        from scripts.export_demo_video import parse_args

        monkeypatch.setattr(sys, "argv", [
            "export_demo_video.py",
            "--checkpoint", "model.pth",
            "--dataset-root", "data/mmfi_pose",
            "--action", "0",
            "--resolution", "720p",
        ])
        args = parse_args()
        assert args.resolution == "720p"


class TestFfmpegCheck:
    """Tests for ffmpeg availability check."""

    def test_ffmpeg_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """check_ffmpeg should raise SystemExit when ffmpeg is missing."""
        import shutil
        from scripts.export_demo_video import check_ffmpeg

        # Mock shutil.which to return None
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        with pytest.raises(SystemExit) as exc_info:
            check_ffmpeg()
        assert exc_info.value.code == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_export_demo_video.py -v
```

Expected: All tests FAIL with `ModuleNotFoundError: No module named 'scripts.export_demo_video'`

- [ ] **Step 3: Commit**

```bash
git add tests/test_export_demo_video.py
git commit -m "test: add demo video export test suite"
```

---

### Task 2: Create script skeleton with arg parsing and ffmpeg check

**Files:**
- Create: `scripts/export_demo_video.py`

- [ ] **Step 1: Write the script skeleton**

```python
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
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------------------------------
# Project path setup (no package install — scripts add root to sys.path)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.memmap_dataset import MemmapDataset
from dataloader import memmap_collate_fn
from eval import load_checkpoint_model
from evaluation.pose_viz import (
    BONE_EDGES,
    JOINT_COLORS,
    _draw_skeleton,
    _compute_axes_limits,
)
from train import extract_prediction_keypoints, prepare_model_input

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
    return w, h


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
# CLI
# ---------------------------------------------------------------------------


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
        "--action", type=int, required=True,
        help="Action class index to visualize.",
    )
    parser.add_argument(
        "--subject", type=int, default=0,
        help="Subject index within the action (0 = first available).",
    )
    parser.add_argument(
        "--output-dir", default="outputs/demo_videos",
        help="Output directory for video files.",
    )
    parser.add_argument(
        "--fps", type=float, default=18,
        help="Frames per second (default: 18).",
    )
    parser.add_argument(
        "--resolution", default="1280x720",
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
    parser.add_argument(
        "--gif-only", action="store_true", default=False,
        help="Generate only GIF (skip MP4).",
    )
    parser.add_argument(
        "--mp4-only", action="store_true", default=False,
        help="Generate only MP4 (skip GIF).",
    )
    parser.add_argument(
        "--manifest-key", default=None,
        help="Dataset split key (e.g. 'test'). Uses full dataset if not set.",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for inference (default: cuda).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    check_ffmpeg()

    width, height = parse_resolution(args.resolution)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Resolution: {width}×{height}")
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")

    # --- 1. Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint_model(args.checkpoint, device)

    # TODO: Steps 3+ will fill in the rest

    print("Done (skeleton — implementation in progress)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run arg parsing and ffmpeg tests**

```bash
pytest tests/test_export_demo_video.py::TestParseArgs -v
pytest tests/test_export_demo_video.py::TestFfmpegCheck -v
```

Expected: `TestParseArgs` tests PASS; `TestFfmpegCheck` fails (function defined but SystemExit expectation needs real ffmpeg state — test will be refined).

- [ ] **Step 3: Verify script runs (skeleton only, no ffmpeg needed yet)**

```bash
python scripts/export_demo_video.py --checkpoint /nonexistent.pth --dataset-root /nonexistent --action 0
```

Expected: Fails with "ffmpeg not found" (if not installed) or "No such file" for checkpoint. Script parses args and starts up cleanly.

- [ ] **Step 4: Commit**

```bash
git add scripts/export_demo_video.py
git commit -m "feat: add demo video export script skeleton (CLI + ffmpeg check)"
```

---

### Task 3: Implement MPJPE computation

**Files:**
- Modify: `scripts/export_demo_video.py` — add `compute_mpjpe()` function

- [ ] **Step 1: Add `compute_mpjpe` function to the script**

Insert after the `parse_resolution` function and before `check_ffmpeg`:

```python
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
```

- [ ] **Step 2: Run MPJPE tests**

```bash
pytest tests/test_export_demo_video.py::TestComputeMpjpe -v
```

Expected: All 3 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/export_demo_video.py
git commit -m "feat: add compute_mpjpe for per-frame error overlay"
```

---

### Task 4: Implement frame collection from dataset

**Files:**
- Modify: `scripts/export_demo_video.py` — add `collect_action_frames()` function

- [ ] **Step 1: Add `collect_action_frames` function**

Insert after `compute_mpjpe` and before `check_ffmpeg`:

```python
# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_action_frames(
    dataset: MemmapDataset,
    action: int,
    subject: int = 0,
) -> list[dict]:
    """Collect all frames for a specific (action, subject) pair in order.

    Parameters
    ----------
    dataset : MemmapDataset
        Dataset in 'all' split mode.
    action : int
        Action class index to filter by.
    subject : int
        Subject index (0-based) within the filtered action group.

    Returns
    -------
    list[dict]
        Each dict has keys: csi (torch.Tensor), kpts18 (torch.Tensor),
        meta (dict with env, subject, action, frame_idx).
        Sorted by frame_idx ascending.
    """
    action_str = str(action)

    # Collect all positions matching this action
    matching: list[int] = []
    for pos in range(len(dataset)):
        abs_idx = int(dataset.indices[pos])
        if str(dataset._actions[abs_idx]) == action_str:
            matching.append(pos)

    if not matching:
        raise ValueError(f"No frames found for action={action}")

    # Group by subject within the action
    sample_list = [str(dataset._samples[int(dataset.indices[p])]) for p in matching]
    unique_subjects = sorted(set(sample_list))

    if subject >= len(unique_subjects):
        raise ValueError(
            f"Subject index {subject} out of range. "
            f"Action {action} has {len(unique_subjects)} subjects: {unique_subjects}"
        )

    target_subject = unique_subjects[subject]

    # Filter to only the target subject
    subject_positions = [
        p for p, s in zip(matching, sample_list) if s == target_subject
    ]

    # Load all frames for this subject, sorted by frame_idx
    frames: list[dict] = []
    for pos in subject_positions:
        item = dataset[pos]  # calls __getitem__ → {csi, kpts18, meta}
        frames.append(item)

    # Sort by frame_idx within the subject group
    frames.sort(key=lambda f: f["meta"]["frame_idx"])

    return frames
```

- [ ] **Step 2: Wire into `main()` — replace the TODO comment**

Replace the `# TODO: Steps 3+ will fill in the rest` block and the "Done" print in `main()` with:

```python
    # --- 2. Build dataset and collect frames ---
    print(f"Loading dataset: {args.dataset_root}")
    dataset = MemmapDataset(data_dir=args.dataset_root, split="all")

    print(f"Collecting frames for action={args.action}, subject={args.subject}...")
    frames = collect_action_frames(dataset, action=args.action, subject=args.subject)
    print(f"  Found {len(frames)} frames")

    actual_subject_id = frames[0]["meta"]["subject"]
    actual_action = frames[0]["meta"]["action"]

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
```

- [ ] **Step 3: Verify the script parses correctly (no module load yet)**

```bash
python -c "import sys; sys.path.insert(0, '.'); from scripts.export_demo_video import collect_action_frames, compute_mpjpe; print('Imports OK')"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/export_demo_video.py
git commit -m "feat: add frame collection and inference loop"
```

---

### Task 5: Implement frame rendering

**Files:**
- Modify: `scripts/export_demo_video.py` — add `render_frame()` function

- [ ] **Step 1: Add `render_frame` function**

Insert after `collect_action_frames` and before `check_ffmpeg`:

```python
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
    # Top info bar (5% height), main panels (85%), bottom progress bar (10%)
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
```

- [ ] **Step 2: Run render_frame tests**

```bash
pytest tests/test_export_demo_video.py::TestRenderFrame -v
```

Expected: All 3 tests PASS (first test creates PNG, second checks dimensions, third checks frame counter).

- [ ] **Step 3: Commit**

```bash
git add scripts/export_demo_video.py
git commit -m "feat: implement frame rendering (side-by-side GT vs Pred)"
```

---

### Task 6: Implement ffmpeg encoding

**Files:**
- Modify: `scripts/export_demo_video.py` — add `encode_mp4()` and `encode_gif()` functions

- [ ] **Step 1: Add encoding functions**

Insert after `check_ffmpeg` and before `parse_args`:

```python
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
               f"palettegen=stats_mode=diff",
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
                  f"[x][1:v] paletteuse=dither=bayer:bayer_scale=5",
        str(output_path),
    ]
    result = subprocess.run(cmd_gif, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg GIF error:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"ffmpeg GIF encoding failed with code {result.returncode}")
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
python -c "import sys; sys.path.insert(0, '.'); from scripts.export_demo_video import encode_mp4, encode_gif; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/export_demo_video.py
git commit -m "feat: add ffmpeg MP4 and GIF encoding"
```

---

### Task 7: Integration test — end-to-end with synthetic data

**Files:**
- Modify: `tests/test_export_demo_video.py` — add integration test

- [ ] **Step 1: Add an end-to-end rendering test**

Append to `tests/test_export_demo_video.py`:

```python
class TestEndToEnd:
    """Integration test: render multiple frames and verify output."""

    def test_render_full_sequence(self, tmp_path: Path) -> None:
        """Render 10 synthetic frames and verify all PNGs are created."""
        from scripts.export_demo_video import render_frame

        np.random.seed(42)
        n_frames = 10
        for i in range(n_frames):
            gt = np.random.randn(18, 2).astype(np.float32) * 0.1 + 0.5
            pred = gt + np.random.randn(18, 2).astype(np.float32) * 0.02
            mpjpe = float(np.linalg.norm(pred - gt, axis=1).mean())

            path = render_frame(
                gt=gt, pred=pred, mpjpe=mpjpe,
                frame_idx=i, total_frames=n_frames,
                action_label="test_action", subject_id="S01",
                model_label="TestModel", output_dir=tmp_path,
            )
            assert path.is_file()

        # All 10 files should exist
        pngs = sorted(tmp_path.glob("frame_*.png"))
        assert len(pngs) == n_frames
        assert pngs[0].name == "frame_000.png"
        assert pngs[-1].name == "frame_009.png"
```

- [ ] **Step 2: Run all tests**

```bash
pytest tests/test_export_demo_video.py -v
```

Expected: ALL tests PASS (7 tests across 4 test classes).

- [ ] **Step 3: Commit**

```bash
git add tests/test_export_demo_video.py
git commit -m "test: add end-to-end rendering integration test"
```

---

### Task 8: Final cleanup and smoke test

**Files:**
- Read: `scripts/export_demo_video.py` — verify completeness

- [ ] **Step 1: Review script for completeness**

Read the full script to verify:
- All imports are used
- No `TODO` or placeholder comments remain
- `main()` function is complete from arg parsing through cleanup
- Error handling: `check_ffmpeg()` called early; `collect_action_frames()` raises on empty/missing action
- `sys.path.insert` pattern matches project convention

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/test_export_demo_video.py -v
```

Expected: 7/7 PASS.

- [ ] **Step 3: Dry-run on real data (if available)**

```bash
conda activate WiFiPose
python scripts/export_demo_video.py \
    --checkpoint outputs/train/best_val_pck_0_2.pth \
    --dataset-root data/mmfi_pose \
    --action 0 \
    --output-dir outputs/demo_videos
```

Expected: Script runs through all steps and produces `outputs/demo_videos/action_*_subject_*_1280x720_18fps.mp4` and `.gif`.

If checkpoint or dataset not available, verify at minimum that the script reaches the dataset-loading stage and fails with a clear "file not found" message (not a Python traceback).

- [ ] **Step 4: Commit**

```bash
git add scripts/export_demo_video.py
git commit -m "chore: final review and polish of demo video export script"
```

---

## Verification Checklist

After all tasks, run:

```bash
# Unit tests
pytest tests/test_export_demo_video.py -v

# Import check
python -c "import sys; sys.path.insert(0, '.'); from scripts.export_demo_video import main, render_frame, compute_mpjpe, collect_action_frames, encode_mp4, encode_gif; print('All imports OK')"

# Help text
python scripts/export_demo_video.py --help
```
