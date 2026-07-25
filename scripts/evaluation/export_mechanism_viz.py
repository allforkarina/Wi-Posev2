from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.split_manifest import load_manifest  # noqa: E402
from eval import build_evaluation_dataset, load_checkpoint_model  # noqa: E402
from evaluation.mechanism_viz import export_mechanism_visualization  # noqa: E402
from train import select_device  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.split_manifest, args.dataset_root)
    device = select_device(args.device)
    model = load_checkpoint_model(
        args.checkpoint,
        device,
        expected_manifest_hash=manifest.manifest_hash,
    )
    dataset = build_evaluation_dataset(
        args.dataset_root,
        manifest=manifest,
        manifest_key=args.manifest_key,
    )
    output = export_mechanism_visualization(
        model,
        dataset,
        device,
        args.output_dir,
        seed=args.seed,
        dpi=args.dpi,
    )
    print(f"Mechanism visualization: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
