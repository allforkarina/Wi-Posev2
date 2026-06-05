from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.feature_viz import _as_axes_grid


def test_as_axes_grid_handles_single_axes_object() -> None:
    fig, axes = plt.subplots(1, 1)

    grid = _as_axes_grid(axes, n_rows=1, n_cols=1)

    assert grid.shape == (1, 1)
    assert grid[0, 0] is axes
    plt.close(fig)


def test_as_axes_grid_handles_single_row_axes_array() -> None:
    fig, axes = plt.subplots(1, 2)

    grid = _as_axes_grid(axes, n_rows=1, n_cols=2)

    assert grid.shape == (1, 2)
    assert grid[0, 0] is axes[0]
    assert grid[0, 1] is axes[1]
    plt.close(fig)
