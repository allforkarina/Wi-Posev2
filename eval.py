"""Evaluate a trained WiFlow model: compute metrics, generate per-category CSVs,
and save CSI/skeleton comparison visualizations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.memmap_dataset import MemmapDataset
from dataloader import create_memmap_data_loader, memmap_collate_fn
from models import WiFlowModel
from train import (
    compute_metrics,
    compute_torso_scale,
    extract_prediction_keypoints,
    prepare_model_input,
    select_device,
)

# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> WiFlowModel:
    """Reconstruct a WiFlowModel from a training checkpoint.

    Reads the saved ``train_config`` dict to restore the correct axial mode
    and decoder type, then loads the learned weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing model_state_dict: {checkpoint_path}")

    train_config = checkpoint.get("train_config")
    if not isinstance(train_config, Mapping):
        raise KeyError(f"Checkpoint is missing train_config: {checkpoint_path}")

    model = WiFlowModel(
        input_channels=3,
        axial_mode=str(train_config.get("axial_mode", "spatial_then_temporal")),
        decoder_type=str(train_config.get("decoder_type", "joint")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-joint error / PCK
# ---------------------------------------------------------------------------


def _joint_errors(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-joint Euclidean distance, shape [B, 18]."""
    return torch.linalg.vector_norm(prediction - target, dim=-1)


def _joint_pck(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-joint PCK boolean mask, shape [B, 18].

    Uses torso diagonal (right shoulder – left hip) as the normalisation
    reference, consistent with the training metric.
    """
    errors = _joint_errors(prediction, target)
    scale = compute_torso_scale(target, eps=eps).unsqueeze(-1)
    return (errors < (scale * threshold)).float()


# ---------------------------------------------------------------------------
# Metric accumulation helpers
# ---------------------------------------------------------------------------


def _update_totals(
    totals: Dict[str, float],
    metrics: Mapping[str, torch.Tensor],
    batch_size: int,
) -> None:
    """Weighted sum of scalar metric tensors into *totals*."""
    for name, value in metrics.items():
        totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size


def _average_metrics(totals: Mapping[str, float], sample_count: int) -> Dict[str, float]:
    """Divide accumulated totals by sample count."""
    return {name: val / max(sample_count, 1) for name, val in totals.items()}


def _update_group_totals(
    totals: Dict[str, Dict[str, float]],
    group_keys: Sequence[str],
    joint_errors: torch.Tensor,
    joint_pck: torch.Tensor,
) -> None:
    """Add one sample's per-joint errors/pck into per-group accumulators."""
    for i, key in enumerate(group_keys):
        entry = totals.setdefault(key, {"count": 0.0, "mpjpe": 0.0, "pck_0_2": 0.0})
        entry["count"] += 1.0
        entry["mpjpe"] += float(joint_errors[i].mean())
        entry["pck_0_2"] += float(joint_pck[i].mean())


def _build_group_rows(
    totals: Mapping[str, Mapping[str, float]],
    group_label: str,
) -> list[dict[str, float | int | str]]:
    """Convert per-group accumulators to a list of dicts (CSV-ready)."""
    rows: list[dict[str, float | int | str]] = []
    for name in sorted(totals):
        entry = totals[name]
        count = int(entry["count"])
        rows.append({
            group_label: name,
            "sample_count": count,
            "mpjpe": entry["mpjpe"] / max(count, 1),
            "pck_0_2": entry["pck_0_2"] / max(count, 1),
        })
    return rows


def _build_joint_rows(
    joint_error_batches: Sequence[torch.Tensor],
    joint_pck_batches: Sequence[torch.Tensor],
) -> list[dict[str, float | int]]:
    """Average per-joint errors/PCK over all samples."""
    all_errors = torch.cat(list(joint_error_batches), dim=0)
    all_pck = torch.cat(list(joint_pck_batches), dim=0)
    total = int(all_errors.shape[0])
    return [
        {
            "joint_index": j,
            "sample_count": total,
            "mpjpe": float(all_errors[:, j].mean()),
            "pck_0_2": float(all_pck[:, j].mean()),
        }
        for j in range(all_errors.shape[1])
    ]


# ---------------------------------------------------------------------------
# Single-pass evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    model: WiFlowModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Run a single forward pass over *loader* and collect all metrics.

    Returns a dict with keys:
    - ``overall``: dict of aggregated metrics (mpjpe, pck_0_1, …).
    - ``joint_rows``: per-joint breakdown (list of dicts).
    - ``action_rows``: per-action breakdown.
    - ``environment_rows``: per-environment breakdown.
    """
    totals: Dict[str, float] = {}
    action_totals: Dict[str, Dict[str, float]] = {}
    environment_totals: Dict[str, Dict[str, float]] = {}
    joint_error_batches: list[torch.Tensor] = []
    joint_pck_batches: list[torch.Tensor] = []
    sample_count = 0

    with torch.no_grad():
        for batch in loader:
            model_input, target = prepare_model_input(batch, device)
            prediction = extract_prediction_keypoints(model(model_input))

            # --- overall metrics (mpjpe, pck_*) ---
            metrics = compute_metrics(prediction, target)
            bs = target.shape[0]
            sample_count += bs
            _update_totals(totals, metrics, bs)

            # --- per-joint & per-group metrics ---
            errors = _joint_errors(prediction, target).detach().cpu()
            pck_mask = _joint_pck(prediction, target).detach().cpu()
            joint_error_batches.append(errors)
            joint_pck_batches.append(pck_mask)
            _update_group_totals(action_totals, batch["action"], errors, pck_mask)
            _update_group_totals(environment_totals, batch["environment"], errors, pck_mask)

    return {
        "overall": _average_metrics(totals, sample_count),
        "joint_rows": _build_joint_rows(joint_error_batches, joint_pck_batches),
        "action_rows": _build_group_rows(action_totals, "action"),
        "environment_rows": _build_group_rows(environment_totals, "environment"),
    }


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a list of homogeneous dicts to a CSV file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained WiFlow pose model.",
    )
    parser.add_argument(
        "--dataset-root", required=True,
        help="Path to the NPY memmap dataset directory.",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a WiFlow checkpoint file.",
    )
    parser.add_argument(
        "--output-dir", default="outputs/eval",
        help="Directory for evaluation CSVs and visualizations.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--eval-envs", nargs="*", default=None,
        help="Filter by environment names (e.g., --eval-envs env1 env2). Evaluates all if not set.",
    )
    parser.add_argument(
        "--exclude-indices", default=None,
        help="Path to .npy file containing frame indices to exclude from evaluation.",
    )
    parser.add_argument(
        "--feature-viz", action="store_true", default=False,
        help="Generate research-grade feature visualization figures.",
    )
    parser.add_argument(
        "--num-action-samples", type=int, default=3,
        help="Samples per action type for feature visualization (default: 3).",
    )
    parser.add_argument(
        "--output-format", choices=["png", "pdf", "both"], default="both",
        help="Output image format for feature visualization (default: both).",
    )
    parser.add_argument(
        "--figure-width", type=float, default=None,
        help="Override default figure width in inches.",
    )
    parser.add_argument(
        "--figure-height", type=float, default=None,
        help="Override default figure height in inches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model = load_checkpoint_model(args.checkpoint, device)

    eval_envs = tuple(args.eval_envs) if args.eval_envs else None
    test_dataset = MemmapDataset(
        data_dir=args.dataset_root,
        split="all",
        envs=eval_envs,
    )

    if args.exclude_indices:
        exclude = np.load(args.exclude_indices)
        exclude_set = set(exclude.tolist())
        keep = [i for i in range(len(test_dataset)) if i not in exclude_set]
        test_dataset = Subset(test_dataset, keep)
        print(f"Excluded {len(exclude_set)} few-shot indices, {len(test_dataset)} remaining")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=memmap_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # --- single-pass evaluation ---
    result = run_evaluation(model, test_loader, device)

    print("--- Test Metrics ---")
    for name in sorted(result["overall"]):
        print(f"  {name}: {result['overall'][name]:.6f}")

    output_dir = Path(args.output_dir)
    _write_csv(output_dir / "per_joint_metrics.csv", result["joint_rows"])
    _write_csv(output_dir / "per_action_metrics.csv", result["action_rows"])
    _write_csv(output_dir / "per_environment_metrics.csv", result["environment_rows"])

    # --- feature visualization (optional, separate pass) ---
    if args.feature_viz:
        from evaluation.feature_viz import run_feature_visualization

        print("\n--- Feature Visualization ---")
        run_feature_visualization(
            model=model,
            loader=test_loader,
            dataset_root=args.dataset_root,
            output_dir=output_dir,
            device=device,
            decoder_type=model.decoder_type,
            num_action_samples=args.num_action_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_format=args.output_format,
            figure_width=args.figure_width,
            figure_height=args.figure_height,
        )
        print("Feature visualization complete.")


if __name__ == "__main__":
    main()