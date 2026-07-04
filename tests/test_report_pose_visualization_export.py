from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.pose_viz import save_pose_comparison


def _pose(offset: float = 0.0) -> np.ndarray:
    values = np.linspace(0.1, 0.9, 36, dtype=np.float32).reshape(18, 2)
    return values + offset


def test_save_pose_comparison_writes_indexed_png_only(tmp_path: Path) -> None:
    path = save_pose_comparison(
        target=_pose(),
        prediction=_pose(0.01),
        action="A01",
        subject="S01",
        environment="env2",
        output_dir=tmp_path,
        dataset_index=123,
        model_label="Finetune 540",
    )

    assert path == tmp_path / "A01" / "S01_env2_idx123.png"
    assert path.is_file()
    assert not list(tmp_path.rglob("*.pdf"))


def test_save_pose_comparison_preserves_legacy_filename_without_index(
    tmp_path: Path,
) -> None:
    path = save_pose_comparison(
        target=_pose(),
        prediction=_pose(0.01),
        action="A01",
        subject="S01",
        environment="env1",
        output_dir=tmp_path,
    )

    assert path == tmp_path / "A01" / "S01_env1.png"


def test_save_pose_comparison_rejects_non_openpose18_shapes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"shape \[18, 2\]"):
        save_pose_comparison(
            target=np.zeros((17, 2), dtype=np.float32),
            prediction=np.zeros((18, 2), dtype=np.float32),
            action="A01",
            subject="S01",
            environment="env1",
            output_dir=tmp_path,
        )
