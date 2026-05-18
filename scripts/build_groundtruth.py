"""
Build merged ground_truth.npy and meta.npz from pre-annotated GT npy files.

Input GT files are located in a flat directory with naming pattern:
    E{env}_S{subject}_A{action}.npy

Each file contains all frames for a single (environment, subject, action) triple:
    shape: (N_frames, 17, 3) float32  ← COCO17 keypoints with (x, y, confidence)

This script:
    1. Reads every E{env}_S{subject}_A{action}.npy
    2. Extracts (x, y) plane from (17, 3) → (17, 2)
    3. Converts COCO17 → OpenPose18 keypoints
    4. Normalizes coordinates to [pose_min, pose_max] range
    5. Concatenates all frames into a single (N, 18, 2) array
    6. Writes ground_truth.npy, meta.npz, and stats.json

Output is drop-in compatible with MemmapDataset.

Usage:
    python scripts/build_groundtruth.py \
        --src /data/WiFiPose/dataset/ground_truth_npy \
        --dst /data/WiFiPose/dataset/mmfi_pose_v3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.heatmap_gt import coco17_to_openpose18, valid_point

POSE_MIN_DEFAULT = -0.8
POSE_MAX_DEFAULT = 0.8
IMG_W = 1920.0
IMG_H = 1080.0


def safe_div(a: float, b: float, eps: float = 1e-6) -> float:
    return a / (b + eps)


def normalize_kpts_to_pose_range(
    kpts: np.ndarray,
    pose_min: float = POSE_MIN_DEFAULT,
    pose_max: float = POSE_MAX_DEFAULT,
) -> np.ndarray:
    kpts = np.asarray(kpts, dtype=np.float32).copy()
    non_zero = kpts[kpts != 0]
    abs_max = float(np.abs(non_zero).max()) if len(non_zero) > 0 else 0.0
    if abs_max > 10.0:
        kpts[..., 0] /= IMG_W
        kpts[..., 1] /= IMG_H
        span = pose_max - pose_min
        kpts = kpts * span + pose_min
    invalid = ~np.isfinite(kpts).all(axis=-1) | np.all(np.isclose(kpts, 0.0), axis=-1)
    kpts[invalid] = 0.0
    kpts = np.clip(kpts, pose_min, pose_max)
    return kpts.astype(np.float32)


FILE_PATTERN = re.compile(r"^E(\d+)_S(\d+)_A(\d+)\.npy$")


def parse_gt_filename(filename: str) -> tuple[str, str, str] | None:
    match = FILE_PATTERN.match(filename)
    if not match:
        return None
    env_num, subj_num, act_num = match.groups()
    return f"E{int(env_num):02d}", f"S{int(subj_num):02d}", f"A{int(act_num):02d}"


def process_gt_file(filepath: Path, pose_min: float, pose_max: float) -> dict | None:
    filename = filepath.name
    parsed = parse_gt_filename(filename)
    if parsed is None:
        return None
    environment, subject, action = parsed

    data = np.load(str(filepath))
    if data.ndim != 3 or data.shape[1:] != (17, 3):
        print(f"  WARNING: {filename} unexpected shape {data.shape}, skipping")
        return None

    coco17_xy = data[:, :, :2].astype(np.float32)

    n_frames = coco17_xy.shape[0]
    kpts18 = np.zeros((n_frames, 18, 2), dtype=np.float32)
    for i in range(n_frames):
        kpts18[i] = normalize_kpts_to_pose_range(
            coco17_to_openpose18(coco17_xy[i]),
            pose_min,
            pose_max,
        )

    return {
        "kpts18": kpts18,
        "environment": environment,
        "sample": subject,
        "action": action,
        "frame_idx": np.arange(1, n_frames + 1, dtype=np.int64),
    }


def main():
    parser = argparse.ArgumentParser(description="Build merged ground_truth.npy from GT npy files")
    parser.add_argument(
        "--src",
        default="/data/WiFiPose/dataset/ground_truth_npy",
        help="Directory containing E{env}_S{subject}_A{action}.npy files",
    )
    parser.add_argument(
        "--dst",
        default="/data/WiFiPose/dataset/mmfi_pose_v3",
        help="Output directory for ground_truth.npy and meta.npz",
    )
    parser.add_argument("--pose-min", type=float, default=POSE_MIN_DEFAULT)
    parser.add_argument("--pose-max", type=float, default=POSE_MAX_DEFAULT)
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.is_dir():
        print(f"ERROR: Source directory not found: {src_dir}")
        sys.exit(1)

    gt_files = sorted(
        p for p in src_dir.iterdir()
        if p.is_file() and p.suffix == ".npy" and parse_gt_filename(p.name) is not None
    )

    if not gt_files:
        print(f"ERROR: No valid GT files found in {src_dir}")
        sys.exit(1)

    print(f"Found {len(gt_files)} GT files in {src_dir}")

    all_kpts18: list[np.ndarray] = []
    all_envs: list[str] = []
    all_subjects: list[str] = []
    all_actions: list[str] = []
    all_fidx: list[np.ndarray] = []
    total_frames = 0
    skipped = 0

    t0 = time.time()
    for i, fp in enumerate(gt_files):
        result = process_gt_file(fp, args.pose_min, args.pose_max)
        if result is None:
            skipped += 1
            continue

        n = result["kpts18"].shape[0]
        all_kpts18.append(result["kpts18"])
        all_envs.extend([result["environment"]] * n)
        all_subjects.extend([result["sample"]] * n)
        all_actions.extend([result["action"]] * n)
        all_fidx.append(result["frame_idx"])
        total_frames += n

        if (i + 1) % 100 == 0 or (i + 1) == len(gt_files):
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{len(gt_files)}] {fp.name} ({n} frames) — {elapsed:.1f}s elapsed")

    if not all_kpts18:
        print("ERROR: No GT files processed successfully")
        sys.exit(1)

    print(f"\nConcatenating {total_frames} frames from {len(all_kpts18)} files ({len(gt_files) - len(all_kpts18)} skipped)...")
    ground_truth = np.concatenate(all_kpts18, axis=0).astype(np.float32)
    envs_arr = np.array(all_envs)
    subjects_arr = np.array(all_subjects)
    actions_arr = np.array(all_actions)
    fidx_arr = np.concatenate(all_fidx)

    print(f"ground_truth: {ground_truth.shape} {ground_truth.dtype}")
    print(f"  x range: [{ground_truth[..., 0].min():.4f}, {ground_truth[..., 0].max():.4f}]")
    print(f"  y range: [{ground_truth[..., 1].min():.4f}, {ground_truth[..., 1].max():.4f}]")

    print("Saving...")
    t_save = time.time()

    np.save(str(dst_dir / "ground_truth.npy"), ground_truth)
    np.savez(
        str(dst_dir / "meta.npz"),
        environment=envs_arr,
        sample=subjects_arr,
        action=actions_arr,
        frame_idx=fidx_arr,
    )

    stats = {
        "total_frames": int(total_frames),
        "total_files": len(all_kpts18),
        "pose_min": args.pose_min,
        "pose_max": args.pose_max,
        "source": str(src_dir.resolve()),
    }
    with open(dst_dir / "gt_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    gt_mb = (dst_dir / "ground_truth.npy").stat().st_size / (1024 * 1024)
    print(f"Done in {time.time() - t_save:.0f}s — ground_truth.npy: {gt_mb:.1f} MB")


if __name__ == "__main__":
    main()