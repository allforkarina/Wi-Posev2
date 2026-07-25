"""Canonical pose schema used throughout Wi-Posev2.

The source arrays follow the Human3.6M 17-joint topology.  The project maps
them into a custom 18-joint index space; it is not COCO or OpenPose ordering.
All training losses, decoder graphs, metrics, and visualizations must import
the definitions in this module instead of maintaining private edge lists.
"""

from __future__ import annotations

from typing import Final

import numpy as np


NUM_RAW_JOINTS: Final = 17
NUM_KEYPOINTS: Final = 18

# output_18_index -> raw_H36M17_index
MAPPED_18_TO_RAW_17: Final[dict[int, int]] = {
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
SYNTHETIC_JOINT_INDEX: Final = 1
SYNTHETIC_JOINT_SOURCES: Final[tuple[int, int]] = (5, 6)

CANONICAL_BONE_EDGES: Final[tuple[tuple[int, int], ...]] = (
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

JOINT_NAMES: Final[tuple[str, ...]] = (
    "pelvis",
    "left_lower_leg_midpoint",
    "left_ankle",
    "thorax",
    "head",
    "left_knee",
    "spine",
    "neck",
    "left_elbow",
    "right_shoulder",
    "right_wrist",
    "left_shoulder",
    "left_wrist",
    "right_elbow",
    "right_knee",
    "right_hip",
    "left_hip",
    "right_ankle",
)

JOINT_GROUPS: Final[dict[str, tuple[int, ...]]] = {
    "torso_head": (0, 3, 4, 6, 7),
    "left_arm": (11, 8, 12),
    "right_arm": (9, 13, 10),
    "left_leg": (16, 5, 1, 2),
    "right_leg": (15, 14, 17),
    "lower_limb": (16, 5, 2, 15, 14, 17),
    "distal": (2, 4, 10, 12, 17),
}

TORSO_DIAGONALS: Final[tuple[tuple[int, int], ...]] = (
    (9, 16),
    (11, 15),
)


def map_raw17_to_project18(raw_xy: np.ndarray) -> np.ndarray:
    """Map ``[..., 17, 2]`` raw coordinates without validity assumptions."""
    values = np.asarray(raw_xy)
    if values.ndim < 2 or values.shape[-2:] != (NUM_RAW_JOINTS, 2):
        raise ValueError(
            f"Expected raw coordinates shaped [..., {NUM_RAW_JOINTS}, 2], "
            f"got {values.shape}"
        )
    mapped = np.empty((*values.shape[:-2], NUM_KEYPOINTS, 2), dtype=values.dtype)
    for mapped_index, raw_index in MAPPED_18_TO_RAW_17.items():
        mapped[..., mapped_index, :] = values[..., raw_index, :]
    mapped[..., SYNTHETIC_JOINT_INDEX, :] = values[
        ..., list(SYNTHETIC_JOINT_SOURCES), :
    ].mean(axis=-2)
    return mapped
