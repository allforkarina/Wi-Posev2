from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval as eval_module
from evaluation.feature_viz import _save_fig


def test_save_fig_writes_png_only(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    output_path = tmp_path / "figure"

    _save_fig(fig, output_path)

    assert output_path.with_suffix(".png").is_file()
    assert not output_path.with_suffix(".pdf").exists()


def test_parse_args_has_no_output_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval.py", "--dataset-root", "dataset", "--checkpoint", "model.pth"],
    )

    args = eval_module.parse_args()

    assert not hasattr(args, "output_format")


def test_parse_args_rejects_legacy_output_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval.py",
            "--dataset-root",
            "dataset",
            "--checkpoint",
            "model.pth",
            "--output-format",
            "pdf",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        eval_module.parse_args()

    assert exc_info.value.code == 2
