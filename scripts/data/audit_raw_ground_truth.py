"""Audit raw 17-joint ground truth before building Wi-Pose memmaps.

This tool is intentionally read-only with respect to the dataset. It records
the raw coordinate distribution, exact ``(0, 0)`` occurrences, non-finite
values, CSI/GT frame-count alignment, and visual previews of the project's
current 17-to-18 mapping.

The audit does not infer that ``(0, 0)`` is invalid and does not rescale,
clip, or otherwise modify coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GT_FILENAME_PATTERN = re.compile(r"^E(\d+)_S(\d+)_A(\d+)\.npy$")

# This is the mapping currently implemented by the repository. Its semantics
# are deliberately not named after COCO, OpenPose, or Human3.6M until the
# visual audit has been reviewed.
CURRENT_18_TO_RAW17: dict[int, int] = {
    0: 0,
    2: 6,
    3: 8,
    4: 10,
    5: 5,
    6: 7,
    7: 9,
    8: 12,
    9: 14,
    10: 16,
    11: 11,
    12: 13,
    13: 15,
    14: 2,
    15: 1,
    16: 4,
    17: 3,
}
SYNTHETIC_JOINT_INDEX = 1
SYNTHETIC_JOINT_SOURCES = (5, 6)

# Confirmed by the project owner as the canonical connectivity in the mapped
# 18-joint index space.
CANONICAL_18_EDGES: tuple[tuple[int, int], ...] = (
    (4, 7),
    (7, 3),
    (3, 9),
    (3, 6),
    (3, 11),
    (9, 13),
    (13, 10),
    (11, 8),
    (8, 12),
    (6, 0),
    (0, 15),
    (0, 16),
    (15, 14),
    (14, 17),
    (16, 5),
    (5, 1),
    (1, 2),
)


@dataclass(frozen=True)
class GroundTruthFile:
    path: Path
    environment: str
    subject: str
    action: str


def parse_ground_truth_file(path: Path) -> GroundTruthFile | None:
    match = GT_FILENAME_PATTERN.match(path.name)
    if match is None:
        return None
    environment, subject, action = (int(value) for value in match.groups())
    return GroundTruthFile(
        path=path,
        environment=f"E{environment:02d}",
        subject=f"S{subject:02d}",
        action=f"A{action:02d}",
    )


def map_current_17_to_18(raw_xy: np.ndarray) -> np.ndarray:
    """Apply the repository's current mapping without validity assumptions."""
    raw_xy = np.asarray(raw_xy)
    if raw_xy.shape != (17, 2):
        raise ValueError(f"Expected one frame shaped (17, 2), got {raw_xy.shape}")
    mapped = np.zeros((18, 2), dtype=raw_xy.dtype)
    for mapped_index, raw_index in CURRENT_18_TO_RAW17.items():
        mapped[mapped_index] = raw_xy[raw_index]
    mapped[SYNTHETIC_JOINT_INDEX] = raw_xy[list(SYNTHETIC_JOINT_SOURCES)].mean(axis=0)
    return mapped


def _quantile_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": int(values.size),
            "finite_count": 0,
            "min": float("nan"),
            "q01": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "q99": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }
    q01, q25, median, q75, q99 = np.quantile(
        finite,
        (0.01, 0.25, 0.5, 0.75, 0.99),
    )
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "min": float(finite.min()),
        "q01": float(q01),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "q99": float(q99),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_indexed_points(
    axis: plt.Axes,
    points: np.ndarray,
    title: str,
    edges: Iterable[tuple[int, int]] = (),
) -> None:
    for start, end in edges:
        axis.plot(
            [points[start, 0], points[end, 0]],
            [points[start, 1], points[end, 1]],
            color="#8a94a6",
            linewidth=1.2,
            zorder=1,
        )
    axis.scatter(points[:, 0], points[:, 1], s=30, color="#2266aa", zorder=2)
    for index, (x_coord, y_coord) in enumerate(points):
        axis.annotate(str(index), (x_coord, y_coord), xytext=(4, 3), textcoords="offset points")
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.2)


def _save_mapping_preview(
    record: GroundTruthFile,
    raw_data: np.ndarray,
    output_dir: Path,
    invert_y: bool,
) -> list[dict[str, Any]]:
    frame_count = int(raw_data.shape[0])
    positions = sorted({0, frame_count // 2, frame_count - 1})
    figure, axes = plt.subplots(len(positions), 2, figsize=(12, 4.5 * len(positions)), squeeze=False)
    rows: list[dict[str, Any]] = []

    for row_index, frame_position in enumerate(positions):
        raw_xy = np.asarray(raw_data[frame_position, :, :2], dtype=np.float64)
        mapped = map_current_17_to_18(raw_xy)
        _plot_indexed_points(
            axes[row_index, 0],
            raw_xy,
            f"Raw 17 joints, frame position {frame_position}",
        )
        _plot_indexed_points(
            axes[row_index, 1],
            mapped,
            f"Current mapped 18 joints, frame position {frame_position}",
            CANONICAL_18_EDGES,
        )
        if invert_y:
            axes[row_index, 0].invert_yaxis()
            axes[row_index, 1].invert_yaxis()
        rows.append({
            "gt_file": record.path.name,
            "frame_position": frame_position,
            "environment": record.environment,
            "subject": record.subject,
            "action": record.action,
        })

    figure.suptitle(f"{record.path.name}: raw-to-current-mapping audit", fontsize=14)
    figure.tight_layout()
    preview_path = output_dir / "mapping_previews" / f"{record.path.stem}.png"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(preview_path, dpi=150)
    plt.close(figure)
    for row in rows:
        row["preview_path"] = str(preview_path)
    return rows


def _select_preview_records(
    records: list[GroundTruthFile],
    count: int,
) -> list[GroundTruthFile]:
    if count <= 0 or not records:
        return []
    if count >= len(records):
        return records
    positions = np.linspace(0, len(records) - 1, count, dtype=int)
    return [records[int(position)] for position in positions]


def audit_ground_truth(
    raw_dataset_root: Path,
    ground_truth_root: Path,
    output_dir: Path,
    preview_file_count: int = 8,
    invert_y: bool = False,
) -> dict[str, Any]:
    """Audit all matching GT files and write compact review artifacts."""
    raw_dataset_root = raw_dataset_root.resolve()
    ground_truth_root = ground_truth_root.resolve()
    output_dir = output_dir.resolve()
    if not raw_dataset_root.is_dir():
        raise FileNotFoundError(f"Raw dataset root not found: {raw_dataset_root}")
    if not ground_truth_root.is_dir():
        raise FileNotFoundError(f"Ground-truth root not found: {ground_truth_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        parsed
        for path in sorted(ground_truth_root.glob("*.npy"))
        if (parsed := parse_ground_truth_file(path)) is not None
    ]
    if not records:
        raise FileNotFoundError(f"No E*_S*_A*.npy files found in {ground_truth_root}")

    coordinate_chunks: list[list[np.ndarray]] = [[], [], []]
    raw_joint_chunks: list[list[np.ndarray]] = [[] for _ in range(17)]
    zero_pair_counts = np.zeros(17, dtype=np.int64)
    finite_pair_counts = np.zeros(17, dtype=np.int64)
    total_joint_observations = np.zeros(17, dtype=np.int64)
    shape_counts: dict[str, int] = {}
    malformed_files: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    valid_records: list[GroundTruthFile] = []
    total_frames = 0

    for record in records:
        data = np.load(record.path, mmap_mode="r")
        shape_key = "x".join(str(value) for value in data.shape)
        shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1
        if data.ndim != 3 or data.shape[1] != 17 or data.shape[2] < 2:
            malformed_files.append({"gt_file": record.path.name, "shape": list(data.shape)})
            continue

        valid_records.append(record)
        frame_count = int(data.shape[0])
        total_frames += frame_count
        numeric = np.asarray(data, dtype=np.float64)
        channel_count = min(int(numeric.shape[2]), 3)
        for channel in range(channel_count):
            coordinate_chunks[channel].append(numeric[:, :, channel].reshape(-1))
        for joint_index in range(17):
            raw_joint_chunks[joint_index].append(numeric[:, joint_index, :2])
            joint_xy = numeric[:, joint_index, :2]
            zero_pair_counts[joint_index] += int(np.all(joint_xy == 0.0, axis=1).sum())
            finite_pair_counts[joint_index] += int(np.isfinite(joint_xy).all(axis=1).sum())
            total_joint_observations[joint_index] += frame_count

        wifi_dir = raw_dataset_root / record.action / record.subject / "wifi-csi"
        csi_frame_count = len(list(wifi_dir.glob("frame*.mat"))) if wifi_dir.is_dir() else 0
        if not wifi_dir.is_dir():
            alignment_status = "missing_wifi_dir"
        elif csi_frame_count != frame_count:
            alignment_status = "frame_count_mismatch"
        else:
            alignment_status = "aligned"
        alignment_rows.append({
            "gt_file": record.path.name,
            "environment": record.environment,
            "subject": record.subject,
            "action": record.action,
            "gt_frames": frame_count,
            "csi_frames": csi_frame_count,
            "status": alignment_status,
            "wifi_dir": str(wifi_dir),
        })

    channel_names = ("channel_0", "channel_1", "channel_2")
    channel_statistics: dict[str, dict[str, float | int]] = {}
    for channel_index, chunks in enumerate(coordinate_chunks):
        if chunks:
            channel_statistics[channel_names[channel_index]] = _quantile_summary(np.concatenate(chunks))

    joint_rows: list[dict[str, Any]] = []
    for joint_index, chunks in enumerate(raw_joint_chunks):
        if not chunks:
            continue
        values = np.concatenate(chunks, axis=0)
        x_stats = _quantile_summary(values[:, 0])
        y_stats = _quantile_summary(values[:, 1])
        observation_count = int(total_joint_observations[joint_index])
        joint_rows.append({
            "raw_joint_index": joint_index,
            "observation_count": observation_count,
            "finite_pair_count": int(finite_pair_counts[joint_index]),
            "exact_zero_pair_count": int(zero_pair_counts[joint_index]),
            "exact_zero_pair_fraction": (
                float(zero_pair_counts[joint_index] / observation_count)
                if observation_count
                else float("nan")
            ),
            "x_min": x_stats["min"],
            "x_median": x_stats["median"],
            "x_max": x_stats["max"],
            "x_mean": x_stats["mean"],
            "x_std": x_stats["std"],
            "y_min": y_stats["min"],
            "y_median": y_stats["median"],
            "y_max": y_stats["max"],
            "y_mean": y_stats["mean"],
            "y_std": y_stats["std"],
        })

    _write_csv(output_dir / "trial_alignment.csv", alignment_rows)
    _write_csv(output_dir / "raw_joint_coordinate_stats.csv", joint_rows)

    preview_rows: list[dict[str, Any]] = []
    for record in _select_preview_records(valid_records, preview_file_count):
        preview_rows.extend(
            _save_mapping_preview(
                record,
                np.load(record.path, mmap_mode="r"),
                output_dir,
                invert_y,
            )
        )
    _write_csv(output_dir / "mapping_preview_manifest.csv", preview_rows)

    alignment_counts: dict[str, int] = {}
    for row in alignment_rows:
        status = str(row["status"])
        alignment_counts[status] = alignment_counts.get(status, 0) + 1

    summary: dict[str, Any] = {
        "schema_version": 1,
        "audit_is_read_only": True,
        "mapping_review_required": True,
        "raw_dataset_root": str(raw_dataset_root),
        "ground_truth_root": str(ground_truth_root),
        "output_dir": str(output_dir),
        "matched_gt_file_count": len(records),
        "valid_gt_file_count": len(valid_records),
        "malformed_gt_file_count": len(malformed_files),
        "total_gt_frames": total_frames,
        "shape_counts": shape_counts,
        "channel_statistics": channel_statistics,
        "raw_joint_exact_zero_pair_counts": {
            str(index): int(count) for index, count in enumerate(zero_pair_counts)
        },
        "alignment_status_counts": alignment_counts,
        "malformed_files": malformed_files,
        "current_mapping": {
            "mapped_18_to_raw_17": {
                str(mapped): raw for mapped, raw in CURRENT_18_TO_RAW17.items()
            },
            "synthetic_joint": {
                "mapped_index": SYNTHETIC_JOINT_INDEX,
                "operation": "mean",
                "raw_source_indices": list(SYNTHETIC_JOINT_SOURCES),
            },
            "verified": False,
        },
        "confirmed_canonical_18_edges": [list(edge) for edge in CANONICAL_18_EDGES],
        "preview_file_count": len(_select_preview_records(valid_records, preview_file_count)),
    }
    with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=True)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit raw 17-joint GT and the current 17-to-18 mapping without modifying data.",
    )
    parser.add_argument("--raw-dataset-root", required=True, type=Path)
    parser.add_argument("--ground-truth-root", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/gt_audit"), type=Path)
    parser.add_argument(
        "--preview-file-count",
        default=8,
        type=int,
        help="Number of GT files sampled across the sorted dataset for mapping previews.",
    )
    parser.add_argument(
        "--invert-y",
        action="store_true",
        help="Invert the y-axis in preview figures if that makes the coordinate convention easier to review.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_ground_truth(
        raw_dataset_root=args.raw_dataset_root,
        ground_truth_root=args.ground_truth_root,
        output_dir=args.output_dir,
        preview_file_count=args.preview_file_count,
        invert_y=args.invert_y,
    )
    print(f"Audited {summary['valid_gt_file_count']} valid GT files.")
    print(f"Total GT frames: {summary['total_gt_frames']}")
    print(f"Alignment status: {summary['alignment_status_counts']}")
    print(f"Review: {Path(summary['output_dir']) / 'audit_summary.json'}")
    print("The current 17-to-18 mapping remains unverified until the preview images are reviewed.")


if __name__ == "__main__":
    main()
