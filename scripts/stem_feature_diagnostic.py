from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.memmap_dataset import MemmapDataset  # noqa: E402
from dataloader import memmap_collate_fn  # noqa: E402
from eval import load_checkpoint_model  # noqa: E402
from train import prepare_model_input, select_device  # noqa: E402


STEM_DIAGNOSTIC_KEYS = (
    "background",
    "residual",
    "raw_feature",
    "residual_feature",
    "fused_feature",
    "gate",
)


def _feature_vector(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.float()
    if x.ndim < 2:
        raise ValueError(f"feature tensor must include a batch dimension, got {tuple(x.shape)}")
    return x.float().flatten(2).mean(dim=2)


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom) < eps:
        return float("nan")
    return float(torch.dot(x, y) / denom)


def _pose_distance_corr(feature: torch.Tensor, keypoints: torch.Tensor) -> float:
    if feature.shape[0] < 3:
        return float("nan")
    feature = _feature_vector(feature)
    pose = keypoints.float().flatten(1)
    feature_distance = torch.cdist(feature, feature, p=2)
    pose_distance = torch.cdist(pose, pose, p=2)
    mask = torch.triu(
        torch.ones(feature.shape[0], feature.shape[0], dtype=torch.bool),
        diagonal=1,
    )
    return _pearson_corr(feature_distance[mask], pose_distance[mask])


def _env_mean_gap(feature: torch.Tensor, envs: Sequence[str]) -> float:
    feature = _feature_vector(feature)
    unique_envs = sorted(set(envs))
    if len(unique_envs) < 2:
        return float("nan")

    means: list[torch.Tensor] = []
    for env in unique_envs:
        mask = torch.as_tensor([item == env for item in envs], dtype=torch.bool)
        if not bool(mask.any()):
            continue
        means.append(feature[mask].mean(dim=0))
    if len(means) < 2:
        return float("nan")

    gaps: list[torch.Tensor] = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            gaps.append(torch.linalg.vector_norm(means[i] - means[j]))
    return float(torch.stack(gaps).mean())


def summarize_stem_diagnostics(
    diagnostics: Mapping[str, torch.Tensor],
    keypoints: torch.Tensor,
    envs: Sequence[str],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if "gate" in diagnostics:
        gate = diagnostics["gate"].float()
        metrics["gate_mean"] = float(gate.mean())
        metrics["gate_std"] = float(gate.std(unbiased=False))
        metrics["gate_min"] = float(gate.min())
        metrics["gate_max"] = float(gate.max())

    for key in STEM_DIAGNOSTIC_KEYS:
        if key not in diagnostics or key == "gate":
            continue
        value = diagnostics[key]
        metrics[f"{key}_env_mean_gap"] = _env_mean_gap(value, envs)
        metrics[f"{key}_pose_distance_corr"] = _pose_distance_corr(value, keypoints)
    return metrics


def collect_stem_diagnostics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[str]]:
    pooled: dict[str, list[torch.Tensor]] = {}
    keypoint_batches: list[torch.Tensor] = []
    envs: list[str] = []
    sample_count = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            model_input, keypoints = prepare_model_input(batch, device)
            _ = model(model_input)
            stem = model.spatial_encoder.stem
            diagnostics = getattr(stem, "latest_diagnostics", {})
            if not diagnostics:
                raise RuntimeError("Spatial stem did not expose latest_diagnostics")

            remaining = max_samples - sample_count
            take = min(model_input.shape[0], remaining)
            for key, value in diagnostics.items():
                if key not in STEM_DIAGNOSTIC_KEYS:
                    continue
                pooled.setdefault(key, []).append(_feature_vector(value[:take].cpu()))
            keypoint_batches.append(keypoints[:take].cpu())
            envs.extend(list(batch["environment"][:take]))
            sample_count += take
            if sample_count >= max_samples:
                break

    if sample_count == 0:
        raise RuntimeError("No samples were collected for stem diagnostics")

    merged = {key: torch.cat(values, dim=0) for key, values in pooled.items()}
    return merged, torch.cat(keypoint_batches, dim=0), envs


def _write_metrics(path: Path, metrics: Mapping[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in sorted(metrics):
            writer.writerow({"metric": key, "value": metrics[key]})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize spatial-stem background/residual diagnostic features.",
    )
    parser.add_argument("--dataset-root", required=True, help="Path to the NPY memmap dataset directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to a WiFlow checkpoint file.")
    parser.add_argument("--output-dir", default="outputs/stem_diagnostic")
    parser.add_argument("--source-envs", nargs="*", default=None)
    parser.add_argument("--target-envs", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")

    envs = None
    selected_envs = []
    if args.source_envs:
        selected_envs.extend(args.source_envs)
    if args.target_envs:
        selected_envs.extend(args.target_envs)
    if selected_envs:
        envs = tuple(selected_envs)

    device = select_device(args.device)
    model = load_checkpoint_model(args.checkpoint, device)
    dataset = MemmapDataset(data_dir=args.dataset_root, split="all", envs=envs)
    if len(dataset) > args.max_samples:
        indices = torch.linspace(0, len(dataset) - 1, steps=args.max_samples).long().tolist()
        dataset = Subset(dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=memmap_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    diagnostics, keypoints, env_labels = collect_stem_diagnostics(
        model=model,
        loader=loader,
        device=device,
        max_samples=args.max_samples,
    )
    metrics = summarize_stem_diagnostics(diagnostics, keypoints, env_labels)
    output_path = Path(args.output_dir) / "stem_diagnostic_metrics.csv"
    _write_metrics(output_path, metrics)

    print("--- Stem Diagnostic Metrics ---")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]:.6f}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
