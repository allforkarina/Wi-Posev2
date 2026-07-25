from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.data.audit_raw_ground_truth import (
    CANONICAL_18_EDGES,
    audit_ground_truth,
    map_current_17_to_18,
)


def test_current_mapping_preserves_zero_as_a_coordinate() -> None:
    raw = np.arange(34, dtype=np.float32).reshape(17, 2)
    raw[5] = 0.0

    mapped = map_current_17_to_18(raw)

    assert mapped.shape == (18, 2)
    np.testing.assert_array_equal(mapped[5], raw[5])
    np.testing.assert_array_equal(mapped[0], raw[0])
    np.testing.assert_allclose(mapped[1], (raw[5] + raw[6]) / 2.0)


def test_audit_reports_alignment_and_writes_json(tmp_path: Path) -> None:
    raw_root = tmp_path / "dataset"
    gt_root = tmp_path / "ground_truth_npy"
    output_dir = tmp_path / "audit"
    wifi_dir = raw_root / "A01" / "S01" / "wifi-csi"
    wifi_dir.mkdir(parents=True)
    gt_root.mkdir()
    for frame_index in range(3):
        (wifi_dir / f"frame{frame_index + 1:03d}.mat").touch()

    ground_truth = np.arange(3 * 17 * 3, dtype=np.float32).reshape(3, 17, 3)
    ground_truth[0, 0, :2] = 0.0
    np.save(gt_root / "E01_S01_A01.npy", ground_truth)

    summary = audit_ground_truth(
        raw_dataset_root=raw_root,
        ground_truth_root=gt_root,
        output_dir=output_dir,
        preview_file_count=0,
    )

    assert summary["alignment_status_counts"] == {"aligned": 1}
    assert summary["raw_joint_exact_zero_pair_counts"]["0"] == 1
    assert summary["current_mapping"]["verified"] is False
    assert summary["confirmed_canonical_18_edges"] == [list(edge) for edge in CANONICAL_18_EDGES]
    saved = json.loads((output_dir / "audit_summary.json").read_text(encoding="utf-8"))
    assert saved["total_gt_frames"] == 3
