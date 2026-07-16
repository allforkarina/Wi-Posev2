from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.split_manifest import load_manifest  # noqa: E402
from dataloader import create_manifest_data_loader  # noqa: E402
from eval import load_checkpoint_model, run_evaluation, write_evaluation_outputs  # noqa: E402
from evaluation.benchmark import (  # noqa: E402
    count_model_operations,
    measure_latency,
    parameter_counts,
    require_device,
    write_runtime_metrics,
)
from train import apply_trainable_groups, prepare_model_input  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one Wi-Pose checkpoint on a fixed manifest split.",
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--measure-iterations", type=int, default=100)
    return parser.parse_args(argv)


def _restore_trainable_group_metadata(
    model: torch.nn.Module,
    checkpoint: Mapping[str, object],
) -> None:
    train_config = checkpoint.get("train_config")
    if not isinstance(train_config, Mapping):
        return
    if train_config.get("mode") != "finetune":
        return
    groups = train_config.get("trainable_groups", ("encoder",))
    apply_trainable_groups(model, tuple(str(group) for group in groups))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = require_device(args.device)
    manifest = load_manifest(args.split_manifest, args.dataset_root)
    model = load_checkpoint_model(
        args.checkpoint,
        device,
        expected_manifest_hash=manifest.manifest_hash,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint, Mapping):
        _restore_trainable_group_metadata(model, checkpoint)

    loader = create_manifest_data_loader(
        data_dir=args.dataset_root,
        manifest=manifest,
        key=args.manifest_key,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    result = run_evaluation(model, loader, device)
    write_evaluation_outputs(args.output_dir, result)

    first_batch = next(iter(loader))
    sample_input, _ = prepare_model_input(first_batch, device)
    sample_input = sample_input[:1]
    operations = count_model_operations(model, sample_input)
    timing = measure_latency(
        model,
        sample_input,
        device=device,
        warmup_iterations=args.warmup_iterations,
        measure_iterations=args.measure_iterations,
    )
    total_parameters, trainable_parameters = parameter_counts(model)
    write_runtime_metrics(
        args.output_dir / "runtime_metrics.csv",
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        operations=operations,
        timing=timing,
        device=device,
        warmup_iterations=args.warmup_iterations,
        measure_iterations=args.measure_iterations,
    )
    print(f"Accuracy and diagnostic metrics: {args.output_dir / 'benchmark_summary.csv'}")
    print(f"Efficiency metrics: {args.output_dir / 'runtime_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
