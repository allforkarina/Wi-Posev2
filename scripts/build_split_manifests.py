from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.split_manifest import (  # noqa: E402
    DatasetMetadata,
    build_split_arrays,
    compute_source_train_normalization,
    save_manifest,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic random-frame and temporal-block split manifests.",
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=16)
    return parser.parse_args(argv)


def _build_one(
    dataset_root: Path,
    output_dir: Path,
    metadata: DatasetMetadata,
    mode: str,
    seed: int,
    block_size: int,
) -> Path:
    arrays = build_split_arrays(
        metadata,
        mode=mode,
        seed=seed,
        block_size=block_size,
    )
    normalization = compute_source_train_normalization(
        dataset_root,
        arrays["env1_train"],
    )
    filename = (
        f"random_frame_seed{seed}.npz"
        if mode == "random_frame"
        else f"temporal_block{block_size}_seed{seed}.npz"
    )
    path = output_dir / filename
    save_manifest(
        path,
        arrays,
        dataset_root=dataset_root,
        mode=mode,
        seed=seed,
        block_size=block_size,
        source_train_normalization=normalization,
    )
    counts = ", ".join(f"{key}={len(values)}" for key, values in arrays.items())
    print(f"Saved {mode} manifest: {path}")
    print(f"  {counts}")
    print(f"  source normalization: min={normalization[0]:.8f}, max={normalization[1]:.8f}")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    meta_path = dataset_root / "meta.npz"
    csi_path = dataset_root / "csi_gminmax.npy"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")
    if not csi_path.is_file():
        raise FileNotFoundError(f"CSI memmap not found: {csi_path}")
    metadata = DatasetMetadata.from_npz(meta_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("random_frame", "temporal_block"):
        _build_one(
            dataset_root=dataset_root,
            output_dir=args.output_dir,
            metadata=metadata,
            mode=mode,
            seed=args.seed,
            block_size=args.block_size,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
