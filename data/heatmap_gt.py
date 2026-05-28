from __future__ import annotations

import numpy as np


COCO17_TO_OPENPOSE18 = {
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


def valid_point(point: np.ndarray) -> bool:
    point = np.asarray(point)
    return bool(np.isfinite(point).all() and not np.allclose(point, 0.0))


def coco17_to_openpose18(kpts17: np.ndarray) -> np.ndarray:
    kpts17 = np.asarray(kpts17, dtype=np.float32)
    if kpts17.shape[-2:] != (17, 2):
        raise ValueError(f"Expected keypoints with shape (17, 2), got {kpts17.shape}")

    kpts18 = np.zeros((18, 2), dtype=np.float32)
    valid = np.zeros(18, dtype=bool)
    for op_idx, coco_idx in COCO17_TO_OPENPOSE18.items():
        point = kpts17[coco_idx]
        if valid_point(point):
            kpts18[op_idx] = point
            valid[op_idx] = True

    left_shoulder = kpts17[5]
    right_shoulder = kpts17[6]
    if valid_point(left_shoulder) and valid_point(right_shoulder):
        kpts18[1] = (left_shoulder + right_shoulder) * 0.5
        valid[1] = True
    elif valid_point(left_shoulder):
        kpts18[1] = left_shoulder
        valid[1] = True
    elif valid_point(right_shoulder):
        kpts18[1] = right_shoulder
        valid[1] = True

    kpts18[~valid] = 0.0
    return kpts18