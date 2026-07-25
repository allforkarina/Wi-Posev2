"""Merge flat audited GT files without rescaling or clipping coordinates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pose_schema import map_raw17_to_project18  # noqa: E402


FILE_PATTERN = re.compile(r"^E(\d+)_S(\d+)_A(\d+)\.npy$")


def parse_gt_filename(filename: str) -> tuple[str, str, str] | None:
    match = FILE_PATTERN.match(filename)
    if match is None:
        return None
    environment, subject, action = (int(value) for value in match.groups())
    return f"env{environment}", f"S{subject:02d}", f"A{action:02d}"


def process_gt_file(filepath: Path) -> dict[str, object] | None:
    parsed = parse_gt_filename(filepath.name)
    if parsed is None:
        return None
    environment, subject, action = parsed
    data = np.load(filepath)
    if data.ndim != 3 or data.shape[1] != 17 or data.shape[2] < 2:
        raise ValueError(f"Expected [N, 17, >=2], got {data.shape}: {filepath}")
    raw_xy = np.asarray(data[..., :2], dtype=np.float32)
    if not np.isfinite(raw_xy).all():
        raise ValueError(f"Non-finite coordinates found: {filepath}")
    return {
        "kpts18": map_raw17_to_project18(raw_xy).astype(np.float32, copy=False),
        "environment": environment,
        "sample": subject,
        "action": action,
        "frame_idx": np.arange(1, len(raw_xy) + 1, dtype=np.int64),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.src.is_dir():
        raise FileNotFoundError(f"GT source directory not found: {args.src}")
    files = sorted(
        path for path in args.src.glob("*.npy")
        if parse_gt_filename(path.name) is not None
    )
    if not files:
        raise FileNotFoundError(f"No valid GT files found in {args.src}")

    results = [process_gt_file(path) for path in files]
    processed = [result for result in results if result is not None]
    ground_truth = np.concatenate(
        [result["kpts18"] for result in processed],
        axis=0,
    ).astype(np.float32, copy=False)
    environments = np.concatenate([
        np.repeat(result["environment"], len(result["kpts18"]))
        for result in processed
    ])
    subjects = np.concatenate([
        np.repeat(result["sample"], len(result["kpts18"]))
        for result in processed
    ])
    actions = np.concatenate([
        np.repeat(result["action"], len(result["kpts18"]))
        for result in processed
    ])
    frame_indices = np.concatenate([
        result["frame_idx"] for result in processed
    ]).astype(np.int64, copy=False)

    args.dst.mkdir(parents=True, exist_ok=True)
    np.save(args.dst / "ground_truth.npy", ground_truth)
    np.savez(
        args.dst / "meta.npz",
        environment=environments,
        sample=subjects,
        action=actions,
        frame_idx=frame_indices,
    )
    (args.dst / "gt_stats.json").write_text(json.dumps({
        "total_frames": len(ground_truth),
        "total_files": len(processed),
        "coordinate_policy": "raw_first_two_channels_no_rescale_no_clip",
        "coordinate_min": ground_truth.min(axis=(0, 1)).tolist(),
        "coordinate_max": ground_truth.max(axis=(0, 1)).tolist(),
        "source": str(args.src.resolve()),
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(ground_truth)} mapped GT frames to {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
