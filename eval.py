"""Evaluate a trained WiFlow model: compute metrics, generate per-category CSVs,
and save CSI/skeleton comparison visualizations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.memmap_dataset import MemmapDataset
from data.pose_schema import (
    CANONICAL_BONE_EDGES,
    JOINT_GROUPS,
    JOINT_NAMES,
    TORSO_DIAGONALS,
)
from data.split_manifest import SplitManifest, load_manifest
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
    expected_manifest_hash: str | None = None,
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
    checkpoint_manifest_hash = train_config.get("manifest_hash")
    if (
        expected_manifest_hash is not None
        and checkpoint_manifest_hash != expected_manifest_hash
    ):
        raise ValueError(
            "Checkpoint manifest hash does not match the requested evaluation manifest"
        )

    model = WiFlowModel(
        input_channels=3,
        axial_mode=str(train_config.get("axial_mode", "spatial_then_temporal")),
        decoder_type=str(train_config.get("decoder_type", "joint")),
        csi_feature_mode=str(train_config.get("csi_feature_mode", "raw")),
        spatial_stem_type=str(train_config.get("spatial_stem_type", "baseline")),
        background_kernel_size=int(train_config.get("background_kernel_size", 9)),
        input_calibration=str(train_config.get("input_calibration", "none")),
        wrist_refinement=bool(train_config.get("wrist_refinement", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_evaluation_dataset(
    dataset_root: str | Path,
    manifest: SplitManifest | None = None,
    manifest_key: str | None = None,
    eval_envs: tuple[str, ...] | None = None,
) -> MemmapDataset:
    if manifest is None:
        if manifest_key is not None:
            raise ValueError("manifest_key requires a split manifest")
        return MemmapDataset(data_dir=dataset_root, split="all", envs=eval_envs)
    if not manifest_key:
        raise ValueError("A manifest key is required for manifest-backed evaluation")
    return MemmapDataset(
        data_dir=dataset_root,
        split="all",
        indices=manifest.indices(manifest_key),
        split_normalization=manifest.source_train_normalization,
    )


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


def _numpy_torso_scale(target: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    diagonals = [
        np.linalg.norm(target[:, start] - target[:, end], axis=-1)
        for start, end in TORSO_DIAGONALS
    ]
    return np.maximum(np.mean(diagonals, axis=0), eps)


def _procrustes_mpjpe(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_center = prediction - prediction.mean(axis=1, keepdims=True)
    target_center = target - target.mean(axis=1, keepdims=True)
    pred_norm = np.maximum(
        np.linalg.norm(pred_center.reshape(len(prediction), -1), axis=1),
        1e-8,
    )
    target_norm = np.maximum(
        np.linalg.norm(target_center.reshape(len(target), -1), axis=1),
        1e-8,
    )
    pred_unit = pred_center / pred_norm[:, None, None]
    target_unit = target_center / target_norm[:, None, None]
    covariance = np.einsum("nji,njk->nik", pred_unit, target_unit)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = np.matmul(left, right_t)
    reflected = np.linalg.det(rotation) < 0
    if reflected.any():
        left[reflected, :, -1] *= -1
        rotation = np.matmul(left, right_t)
    aligned = (
        np.matmul(pred_unit, rotation) * target_norm[:, None, None]
        + target.mean(axis=1, keepdims=True)
    )
    return float(np.linalg.norm(aligned - target, axis=-1).mean())


def _array_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if len(prediction) == 0:
        raise ValueError("Cannot compute metrics for an empty evaluation subset")
    errors = np.linalg.norm(prediction - target, axis=-1)
    coordinate_delta = np.abs(prediction - target)
    scale = _numpy_torso_scale(target)
    normalized_error = errors / scale[:, None]
    pck_thresholds = np.linspace(0.0, 0.5, 101)
    pck_curve = np.asarray([
        np.mean(normalized_error < threshold)
        for threshold in pck_thresholds
    ])

    flattened_prediction = prediction.reshape(len(prediction), -1)
    flattened_target = target.reshape(len(target), -1)
    scale_factor = (
        np.sum(flattened_prediction * flattened_target, axis=1)
        / np.maximum(np.sum(flattened_prediction ** 2, axis=1), 1e-8)
    )
    scale_aligned = prediction * scale_factor[:, None, None]
    root_prediction = prediction - prediction[:, :1]
    root_target = target - target[:, :1]

    edge_index = np.asarray(CANONICAL_BONE_EDGES, dtype=np.int64)
    pred_bones = (
        prediction[:, edge_index[:, 1]] - prediction[:, edge_index[:, 0]]
    )
    target_bones = target[:, edge_index[:, 1]] - target[:, edge_index[:, 0]]
    pred_lengths = np.linalg.norm(pred_bones, axis=-1)
    target_lengths = np.linalg.norm(target_bones, axis=-1)
    bone_absolute = np.abs(pred_lengths - target_lengths)
    bone_relative = bone_absolute / np.maximum(target_lengths, 1e-6)
    cosine = np.sum(pred_bones * target_bones, axis=-1) / np.maximum(
        pred_lengths * target_lengths,
        1e-8,
    )
    identical_bones = np.all(
        np.isclose(pred_bones, target_bones, rtol=0.0, atol=1e-8),
        axis=-1,
    )
    cosine = np.where(identical_bones, 1.0, cosine)
    bone_angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    symmetric_paths = (
        ((16, 5), (15, 14)),
        ((5, 2), (14, 17)),
        ((11, 8), (9, 13)),
        ((8, 12), (13, 10)),
    )
    symmetry_errors = []
    for left_edge, right_edge in symmetric_paths:
        pred_difference = (
            np.linalg.norm(
                prediction[:, left_edge[0]] - prediction[:, left_edge[1]],
                axis=-1,
            )
            - np.linalg.norm(
                prediction[:, right_edge[0]] - prediction[:, right_edge[1]],
                axis=-1,
            )
        )
        target_difference = (
            np.linalg.norm(
                target[:, left_edge[0]] - target[:, left_edge[1]],
                axis=-1,
            )
            - np.linalg.norm(
                target[:, right_edge[0]] - target[:, right_edge[1]],
                axis=-1,
            )
        )
        symmetry_errors.append(np.abs(pred_difference - target_difference))

    invalid_skeleton = (
        ~np.isfinite(prediction).all(axis=(1, 2))
        | (pred_lengths < 1e-6).any(axis=1)
    )
    return {
        "mpjpe": float(errors.mean()),
        "median_joint_error": float(np.median(errors)),
        "p90_joint_error": float(np.percentile(errors, 90)),
        "p95_joint_error": float(np.percentile(errors, 95)),
        "coordinate_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "x_mae": float(coordinate_delta[..., 0].mean()),
        "y_mae": float(coordinate_delta[..., 1].mean()),
        "n_mpjpe": float(np.linalg.norm(scale_aligned - target, axis=-1).mean()),
        "root_relative_mpjpe": float(
            np.linalg.norm(root_prediction - root_target, axis=-1).mean()
        ),
        "pa_mpjpe": _procrustes_mpjpe(prediction, target),
        "bone_error": float(bone_absolute.mean()),
        "relative_bone_length_error": float(bone_relative.mean()),
        "bone_direction_error_deg": float(bone_angle.mean()),
        "symmetry_error": float(np.mean(symmetry_errors)),
        "invalid_skeleton_rate": float(invalid_skeleton.mean()),
        "pck_0_05": float(np.mean(normalized_error < 0.05)),
        "pck_0_1": float(np.mean(normalized_error < 0.1)),
        "pck_0_2": float(np.mean(normalized_error < 0.2)),
        "pck_0_3": float(np.mean(normalized_error < 0.3)),
        "pck_0_4": float(np.mean(normalized_error < 0.4)),
        "pck_0_5": float(np.mean(normalized_error < 0.5)),
        "pck_auc_0_5": float(np.trapz(pck_curve, pck_thresholds) / 0.5),
    }


def _joint_metric_rows(
    prediction: np.ndarray,
    target: np.ndarray,
) -> list[dict[str, float | int | str]]:
    errors = np.linalg.norm(prediction - target, axis=-1)
    normalized = errors / _numpy_torso_scale(target)[:, None]
    return [
        {
            "joint_index": joint_index,
            "joint_name": JOINT_NAMES[joint_index],
            "sample_count": len(prediction),
            "mpjpe": float(errors[:, joint_index].mean()),
            "median_error": float(np.median(errors[:, joint_index])),
            "p90_error": float(np.percentile(errors[:, joint_index], 90)),
            "p95_error": float(np.percentile(errors[:, joint_index], 95)),
            "pck_0_2": float(np.mean(normalized[:, joint_index] < 0.2)),
        }
        for joint_index in range(errors.shape[1])
    ]


def _joint_group_rows(
    prediction: np.ndarray,
    target: np.ndarray,
) -> list[dict[str, float | int | str]]:
    errors = np.linalg.norm(prediction - target, axis=-1)
    normalized = errors / _numpy_torso_scale(target)[:, None]
    rows: list[dict[str, float | int | str]] = []
    for group_name, joints in JOINT_GROUPS.items():
        group_errors = errors[:, joints]
        rows.append({
            "joint_group": group_name,
            "joint_indices": " ".join(str(index) for index in joints),
            "sample_count": len(prediction),
            "mpjpe": float(group_errors.mean()),
            "median_error": float(np.median(group_errors)),
            "p90_error": float(np.percentile(group_errors, 90)),
            "pck_0_2": float(np.mean(normalized[:, joints] < 0.2)),
        })
    return rows


def _category_rows(
    labels: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    label_name: str,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        metrics = _array_metrics(prediction[mask], target[mask])
        rows.append({
            label_name: str(label),
            "sample_count": int(mask.sum()),
            **metrics,
        })
    return rows


def _temporal_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    environments: np.ndarray,
    subjects: np.ndarray,
    actions: np.ndarray,
    frame_indices: np.ndarray,
) -> dict[str, float | int]:
    grouped: dict[tuple[str, str, str], list[int]] = {}
    for index, key in enumerate(zip(environments, subjects, actions)):
        grouped.setdefault(tuple(str(value) for value in key), []).append(index)
    velocity_errors: list[np.ndarray] = []
    acceleration_errors: list[np.ndarray] = []
    for indices in grouped.values():
        ordered = sorted(indices, key=lambda value: int(frame_indices[value]))
        if len(ordered) < 2:
            continue
        for left, right in zip(ordered[:-1], ordered[1:]):
            delta = int(frame_indices[right]) - int(frame_indices[left])
            if delta <= 0:
                continue
            pred_velocity = (prediction[right] - prediction[left]) / delta
            target_velocity = (target[right] - target[left]) / delta
            velocity_errors.append(
                np.linalg.norm(pred_velocity - target_velocity, axis=-1)
            )
        for first, middle, last in zip(ordered[:-2], ordered[1:-1], ordered[2:]):
            delta_one = int(frame_indices[middle]) - int(frame_indices[first])
            delta_two = int(frame_indices[last]) - int(frame_indices[middle])
            if delta_one <= 0 or delta_two <= 0:
                continue
            pred_v1 = (prediction[middle] - prediction[first]) / delta_one
            pred_v2 = (prediction[last] - prediction[middle]) / delta_two
            target_v1 = (target[middle] - target[first]) / delta_one
            target_v2 = (target[last] - target[middle]) / delta_two
            time_step = (delta_one + delta_two) / 2.0
            acceleration_errors.append(
                np.linalg.norm(
                    (pred_v2 - pred_v1) / time_step
                    - (target_v2 - target_v1) / time_step,
                    axis=-1,
                )
            )
    return {
        "temporal_velocity_error": (
            float(np.concatenate(velocity_errors).mean())
            if velocity_errors else float("nan")
        ),
        "temporal_acceleration_error": (
            float(np.concatenate(acceleration_errors).mean())
            if acceleration_errors else float("nan")
        ),
        "temporal_pair_count": len(velocity_errors),
        "temporal_triplet_count": len(acceleration_errors),
    }


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
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_actions: list[str] = []
    all_environments: list[str] = []
    all_subjects: list[str] = []
    all_frame_indices: list[int] = []

    with torch.no_grad():
        for batch in loader:
            model_input, target = prepare_model_input(batch, device)
            prediction = extract_prediction_keypoints(model(model_input))
            all_predictions.append(prediction.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
            all_actions.extend(str(value) for value in batch["action"])
            all_environments.extend(str(value) for value in batch["environment"])
            all_subjects.extend(str(value) for value in batch["sample"])
            all_frame_indices.extend(int(value) for value in batch["frame_idx"])

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    actions = np.asarray(all_actions)
    environments = np.asarray(all_environments)
    subjects = np.asarray(all_subjects)
    frame_indices = np.asarray(all_frame_indices, dtype=np.int64)
    overall = _array_metrics(predictions, targets)
    overall.update(_temporal_metrics(
        predictions,
        targets,
        environments,
        subjects,
        actions,
        frame_indices,
    ))

    return {
        "sample_count": len(predictions),
        "overall": overall,
        "joint_rows": _joint_metric_rows(predictions, targets),
        "joint_group_rows": _joint_group_rows(predictions, targets),
        "action_rows": _category_rows(
            actions,
            predictions,
            targets,
            "action",
        ),
        "environment_rows": _category_rows(
            environments,
            predictions,
            targets,
            "environment",
        ),
        "diagnostic": _compute_diagnostics(all_predictions, all_targets),
    }


# ---------------------------------------------------------------------------
# Mean-pose collapse diagnostics
# ---------------------------------------------------------------------------


def _compute_diagnostics(
    all_predictions: Sequence[np.ndarray],
    all_targets: Sequence[np.ndarray],
) -> Dict[str, Any]:
    """Compute per-joint variance and mean-pose distance.

    Parameters
    ----------
    all_predictions : list of ndarray, each [B, 18, 2]
    all_targets : list of ndarray, each [B, 18, 2]

    Returns
    -------
    dict with ``overall`` averaged metrics and ``joint_rows`` list of dicts.
    """
    preds = np.concatenate(list(all_predictions), axis=0)  # [N, 18, 2]
    targets = np.concatenate(list(all_targets), axis=0)    # [N, 18, 2]

    # per-joint variance over sample axis, averaged over x/y
    pred_var = preds.var(axis=0).mean(axis=1)   # [18]
    gt_var = targets.var(axis=0).mean(axis=1)    # [18]
    var_ratio = np.divide(
        pred_var,
        gt_var,
        out=np.zeros_like(pred_var),
        where=gt_var > 1e-8,
    )
    pred_std = np.sqrt(pred_var)
    gt_std = np.sqrt(gt_var)
    std_ratio = np.divide(
        pred_std,
        gt_std,
        out=np.zeros_like(pred_std),
        where=gt_std > 1e-8,
    )

    # L2 distance between per-joint means
    pred_mean = preds.mean(axis=0)   # [18, 2]
    gt_mean = targets.mean(axis=0)   # [18, 2]
    mean_pose_dist = np.linalg.norm(pred_mean - gt_mean, axis=1)  # [18]

    joint_rows = [
        {
            "joint_index": j,
            "joint_name": JOINT_NAMES[j],
            "joint_groups": " ".join(
                name for name, indices in JOINT_GROUPS.items() if j in indices
            ),
            "pred_std": float(pred_std[j]),
            "gt_std": float(gt_std[j]),
            "std_ratio": float(std_ratio[j]),
            "pred_var": float(pred_var[j]),
            "gt_var": float(gt_var[j]),
            "var_ratio": float(var_ratio[j]),
            "mean_pose_dist": float(mean_pose_dist[j]),
        }
        for j in range(18)
    ]

    overall = {
        "overall_pred_var": float(pred_var.mean()),
        "overall_gt_var": float(gt_var.mean()),
        "overall_var_ratio": float(var_ratio.mean()),
        "overall_pred_std": float(pred_std.mean()),
        "overall_gt_std": float(gt_std.mean()),
        "overall_std_ratio": float(std_ratio.mean()),
        "overall_mean_pose_dist": float(mean_pose_dist.mean()),
    }

    return {"overall": overall, "joint_rows": joint_rows}


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


def write_evaluation_outputs(output_dir: str | Path, result: Mapping[str, Any]) -> None:
    output_dir = Path(output_dir)
    diagnostic_overall = result["diagnostic"]["overall"]
    summary = {
        "sample_count": int(result["sample_count"]),
        **result["overall"],
        "overall_var_ratio": float(diagnostic_overall["overall_var_ratio"]),
        "overall_std_ratio": float(diagnostic_overall["overall_std_ratio"]),
        "overall_mean_pose_dist": float(diagnostic_overall["overall_mean_pose_dist"]),
    }
    _write_csv(output_dir / "benchmark_summary.csv", [summary])
    _write_csv(output_dir / "per_joint_metrics.csv", result["joint_rows"])
    _write_csv(output_dir / "per_joint_group_metrics.csv", result["joint_group_rows"])
    _write_csv(output_dir / "per_action_metrics.csv", result["action_rows"])
    _write_csv(output_dir / "per_environment_metrics.csv", result["environment_rows"])
    _write_csv(output_dir / "per_joint_diagnostic.csv", result["diagnostic"]["joint_rows"])


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
        "--split-manifest",
        default=None,
        help="Path to a deterministic split manifest.",
    )
    parser.add_argument(
        "--manifest-key",
        default=None,
        help="Named array in --split-manifest, such as env1_test or env2_test.",
    )
    parser.add_argument(
        "--pose-viz", action="store_true", default=False,
        help="Generate per-subject joint scatter plots (GT vs Prediction).",
    )
    parser.add_argument(
        "--figure-width", type=float, default=None,
        help="Override default figure width in inches.",
    )
    parser.add_argument(
        "--figure-height", type=float, default=None,
        help="Override default figure height in inches.",
    )
    parser.add_argument(
        "--pose-viz-sampling",
        choices=("random", "middle"),
        default="random",
    )
    parser.add_argument("--pose-viz-seed", type=int, default=42)
    parser.add_argument("--pose-viz-max-subjects-per-action", type=int, default=2)
    parser.add_argument("--pose-viz-dpi", type=int, default=150)
    parser.add_argument(
        "--pose-viz-include-individuals",
        action="store_true",
        help="Also save per-sample figures; composites alone are the default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.split_manifest) != bool(args.manifest_key):
        raise ValueError("--split-manifest and --manifest-key must be provided together")
    eval_envs = tuple(args.eval_envs) if args.eval_envs else None
    manifest = (
        load_manifest(args.split_manifest, args.dataset_root)
        if args.split_manifest
        else None
    )
    device = select_device(args.device)
    model = load_checkpoint_model(
        args.checkpoint,
        device,
        expected_manifest_hash=manifest.manifest_hash if manifest else None,
    )
    test_dataset = build_evaluation_dataset(
        dataset_root=args.dataset_root,
        manifest=manifest,
        manifest_key=args.manifest_key,
        eval_envs=eval_envs,
    )

    if args.exclude_indices:
        exclude = np.load(args.exclude_indices)
        exclude_set = set(exclude.tolist())
        keep = np.asarray([
            int(index)
            for index in test_dataset.indices
            if int(index) not in exclude_set
        ], dtype=np.int64)
        test_dataset = MemmapDataset(
            data_dir=args.dataset_root,
            split="all",
            indices=keep,
            split_normalization=(
                manifest.source_train_normalization if manifest else None
            ),
        )
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
    write_evaluation_outputs(output_dir, result)

    print("\n--- Diagnostic Metrics (mean-pose collapse) ---")
    for name in sorted(result["diagnostic"]["overall"]):
        print(f"  {name}: {result['diagnostic']['overall'][name]:.6f}")
    print(f"  (var_ratio < 0.3 strongly suggests mean-pose collapse)")

    # --- pose visualization (optional, separate pass) ---
    if args.pose_viz:
        from evaluation.pose_viz import run_pose_visualization

        print("\n--- Pose Joint Scatter Visualization ---")
        run_pose_visualization(
            model=model,
            dataset=test_dataset,
            device=device,
            output_dir=output_dir,
            figure_width=args.figure_width,
            figure_height=args.figure_height,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampling=args.pose_viz_sampling,
            seed=args.pose_viz_seed,
            max_subjects_per_action=args.pose_viz_max_subjects_per_action,
            composite_only=not args.pose_viz_include_individuals,
            dpi=args.pose_viz_dpi,
        )
        print("Pose visualization complete.")


if __name__ == "__main__":
    main()
