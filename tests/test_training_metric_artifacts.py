from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import append_epoch_metric_csvs, compute_metrics


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def test_compute_metrics_reports_bone_error() -> None:
    target = torch.zeros(1, 18, 2)
    prediction = target.clone()
    prediction[:, 1, 0] = 1.0

    metrics = compute_metrics(prediction, target)

    assert metrics["bone_error"].item() > 0.0


def test_append_epoch_metric_csvs_separates_metric_families(tmp_path: Path) -> None:
    train_metrics = {
        "loss": 1.0,
        "coord_loss": 0.4,
        "bone_loss": 0.2,
        "source_loss": 0.1,
        "mpjpe": 0.3,
        "bone_error": 0.2,
        "pck_0_05": 0.05,
        "pck_0_1": 0.1,
        "pck_0_2": 0.2,
        "pck_0_3": 0.3,
        "pck_0_4": 0.4,
        "pck_0_5": 0.5,
    }
    val_metrics = {
        "loss": 0.8,
        "coord_loss": 0.3,
        "bone_loss": 0.15,
        "mpjpe": 0.25,
        "bone_error": 0.15,
        "pck_0_05": 0.075,
        "pck_0_1": 0.15,
        "pck_0_2": 0.25,
        "pck_0_3": 0.35,
        "pck_0_4": 0.45,
        "pck_0_5": 0.55,
    }

    append_epoch_metric_csvs(tmp_path, 1, train_metrics, val_metrics)
    append_epoch_metric_csvs(tmp_path, 2, train_metrics, val_metrics)

    expected_paths = {
        "loss.csv",
        "mpjpe.csv",
        "bone_error.csv",
        "pck_0_05.csv",
        "pck_0_1.csv",
        "pck_0_2.csv",
        "pck_0_3.csv",
        "pck_0_4.csv",
        "pck_0_5.csv",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_paths

    loss_rows = _read_rows(tmp_path / "loss.csv")
    assert [row["epoch"] for row in loss_rows] == ["1", "2"]
    assert float(loss_rows[0]["train_source_loss"]) == pytest.approx(0.1)
    assert float(loss_rows[0]["val_loss"]) == pytest.approx(0.8)

    mpjpe_rows = _read_rows(tmp_path / "mpjpe.csv")
    assert set(mpjpe_rows[0]) == {"epoch", "train_mpjpe", "val_mpjpe"}
    assert "train_source_loss" not in mpjpe_rows[0]

    for threshold in ("05", "1", "2", "3", "4", "5"):
        metric_name = f"pck_0_{threshold}"
        rows = _read_rows(tmp_path / f"{metric_name}.csv")
        assert len(rows) == 2
        assert set(rows[0]) == {"epoch", f"train_{metric_name}", f"val_{metric_name}"}


def test_append_epoch_metric_csvs_supports_finetune_without_validation(tmp_path: Path) -> None:
    train_metrics = {
        "loss": 1.0,
        "coord_loss": 0.4,
        "bone_loss": 0.2,
        "target_loss": 0.9,
        "mpjpe": 0.3,
        "bone_error": 0.2,
        "pck_0_05": 0.05,
        "pck_0_1": 0.1,
        "pck_0_2": 0.2,
        "pck_0_3": 0.3,
        "pck_0_4": 0.4,
        "pck_0_5": 0.5,
    }

    append_epoch_metric_csvs(tmp_path, 1, train_metrics)

    assert set(_read_rows(tmp_path / "mpjpe.csv")[0]) == {"epoch", "train_mpjpe"}
    assert set(_read_rows(tmp_path / "loss.csv")[0]) == {
        "epoch",
        "train_loss",
        "train_coord_loss",
        "train_bone_loss",
        "train_target_loss",
    }
