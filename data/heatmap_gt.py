"""Backward-compatible accessors for the project pose mapping."""

from __future__ import annotations

import numpy as np

from data.pose_schema import MAPPED_18_TO_RAW_17, map_raw17_to_project18


# Retained import alias for older callers. The source is H36M17-like and the
# destination is the project-specific 18-joint ordering, not COCO/OpenPose.
COCO17_TO_OPENPOSE18 = MAPPED_18_TO_RAW_17


def valid_point(point: np.ndarray) -> bool:
    """Return whether a coordinate is finite; ``(0, 0)`` remains valid."""
    return bool(np.isfinite(np.asarray(point)).all())


def coco17_to_openpose18(kpts17: np.ndarray) -> np.ndarray:
    """Compatibility wrapper around :func:`map_raw17_to_project18`."""
    return map_raw17_to_project18(
        np.asarray(kpts17, dtype=np.float32)
    ).astype(np.float32, copy=False)
