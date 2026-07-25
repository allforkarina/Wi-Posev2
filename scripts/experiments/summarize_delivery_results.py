"""Aggregate multi-seed delivery metrics and paired ablation effects."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


IDENTITY_FIELDS = {
    "seed",
    "experiment_id",
    "local_id",
    "split_mode",
    "phase",
    "manifest_key",
}
FULL_EXPERIMENT = "random_frame_AX6_JD3_C3"
T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}
ABLATION_CONTROLS = (
    "random_frame_AX0_C2",
    "random_frame_AX1",
    "random_frame_AX2",
    "random_frame_AX3",
    "random_frame_AX4",
    "random_frame_AX5",
    "random_frame_JD0_C1",
    "random_frame_JD1",
    "random_frame_JD2",
    "random_frame_JD4",
    "random_frame_JD5",
    "random_frame_C0",
)


def _read_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def _numeric_metrics(rows: Sequence[dict[str, str]]) -> tuple[str, ...]:
    if not rows:
        return ()
    metrics: list[str] = []
    for name in rows[0]:
        if name in IDENTITY_FIELDS:
            continue
        try:
            for row in rows:
                float(row[name])
        except (KeyError, TypeError, ValueError):
            continue
        metrics.append(name)
    return tuple(metrics)


def _mean_std(values: Sequence[float]) -> tuple[float, float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    degrees_of_freedom = len(values) - 1
    critical = T_975.get(degrees_of_freedom, 1.96)
    return mean, std, critical * std / math.sqrt(len(values))


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: Sequence[dict[str, str]]) -> list[dict[str, object]]:
    metrics = _numeric_metrics(rows)
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["experiment_id"],
            row["local_id"],
            row["split_mode"],
            row["phase"],
            row["manifest_key"],
        )
        grouped[key].append(row)
    output: list[dict[str, object]] = []
    for key, group_rows in sorted(grouped.items()):
        base: dict[str, object] = {
            "experiment_id": key[0],
            "local_id": key[1],
            "split_mode": key[2],
            "phase": key[3],
            "manifest_key": key[4],
            "seed_count": len(group_rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group_rows]
            mean, std, ci95 = _mean_std(values)
            base[f"{metric}_mean"] = mean
            base[f"{metric}_std"] = std
            base[f"{metric}_ci95"] = ci95
        output.append(base)
    return output


def paired_ablation_effects(
    rows: Sequence[dict[str, str]],
) -> list[dict[str, object]]:
    indexed = {
        (row["experiment_id"], row["manifest_key"], int(row["seed"])): row
        for row in rows
    }
    output: list[dict[str, object]] = []
    for control in ABLATION_CONTROLS:
        for manifest_key in ("env1_test", "env2_test"):
            mpjpe_deltas: list[float] = []
            pck_deltas: list[float] = []
            for seed in (42, 123, 3407):
                full = indexed.get((FULL_EXPERIMENT, manifest_key, seed))
                baseline = indexed.get((control, manifest_key, seed))
                if full is None or baseline is None:
                    continue
                mpjpe_deltas.append(float(baseline["mpjpe"]) - float(full["mpjpe"]))
                pck_deltas.append(float(full["pck_0_2"]) - float(baseline["pck_0_2"]))
            if not mpjpe_deltas:
                continue
            mpjpe_mean, mpjpe_std, mpjpe_ci95 = _mean_std(mpjpe_deltas)
            pck_mean, pck_std, pck_ci95 = _mean_std(pck_deltas)
            output.append({
                "full_experiment": FULL_EXPERIMENT,
                "control_experiment": control,
                "manifest_key": manifest_key,
                "paired_seed_count": len(mpjpe_deltas),
                "mpjpe_improvement_mean": mpjpe_mean,
                "mpjpe_improvement_std": mpjpe_std,
                "mpjpe_improvement_ci95": mpjpe_ci95,
                "pck_0_2_improvement_mean": pck_mean,
                "pck_0_2_improvement_std": pck_std,
                "pck_0_2_improvement_ci95": pck_ci95,
            })
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.output_root.resolve()
    index_paths = sorted(root.glob("seed*/evaluation_index.csv"))
    if not index_paths:
        raise FileNotFoundError(f"No seed*/evaluation_index.csv files found under {root}")
    rows = _read_rows(index_paths)
    aggregate_rows = aggregate(rows)
    effect_rows = paired_ablation_effects(rows)
    summary_dir = root / "summary"
    _write_csv(summary_dir / "all_seed_evaluations.csv", rows)
    _write_csv(summary_dir / "aggregate_metrics.csv", aggregate_rows)
    _write_csv(summary_dir / "paired_ablation_effects.csv", effect_rows)
    (summary_dir / "summary.json").write_text(json.dumps({
        "seed_indexes": [str(path) for path in index_paths],
        "evaluation_row_count": len(rows),
        "aggregate_row_count": len(aggregate_rows),
        "paired_ablation_row_count": len(effect_rows),
        "checkpoint_selection": "minimum_validation_mpjpe",
        "positive_mpjpe_improvement": "control_mpjpe - full_mpjpe",
        "positive_pck_improvement": "full_pck - control_pck",
    }, indent=2), encoding="utf-8")
    print(f"Multi-seed summary: {summary_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
