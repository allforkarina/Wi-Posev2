from __future__ import annotations

"""NPY memmap-backed dataloader for MM-Fi pose data."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader

from data.memmap_dataset import MemmapDataset

SPLIT_NAMES = ("train", "val", "test")


def memmap_collate_fn(batch: list[dict]) -> dict:
    csi = torch.stack([item["csi"] for item in batch])
    csi = csi.permute(0, 2, 3, 1).contiguous()
    keypoints = torch.stack([item["kpts18"] for item in batch])
    return {
        "csi_amplitude": csi,
        "keypoints": keypoints,
        "action": [item["meta"]["action"] for item in batch],
        "sample": [item["meta"]["subject"] for item in batch],
        "environment": [item["meta"]["env"] for item in batch],
        "frame_idx": [item["meta"]["frame_idx"] for item in batch],
    }


def create_memmap_data_loader(
    data_dir: str | Path,
    split: str,
    batch_size: int,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    seed: int = 42,
) -> DataLoader:
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split}")

    dataset = MemmapDataset(
        data_dir=data_dir,
        split=split,
        seed=seed,
        build_targets=False,
    )
    should_shuffle = shuffle if shuffle is not None else split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        collate_fn=memmap_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def create_memmap_data_loaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, DataLoader]:
    return {
        split: create_memmap_data_loader(
            data_dir=data_dir,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )
        for split in SPLIT_NAMES
    }


def create_da_data_loaders(
    data_dir: str | Path,
    source_envs: Sequence[str],
    target_envs: Sequence[str],
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Create DataLoaders for domain-adaptation training.

    Source domain uses all filtered data (``split="all"``, no val split).
    Target domain is split into train / val / test by subject (80/20).

    Returns a dict with keys:
      ``"source_train"``, ``"target_train"``, ``"target_val"``, ``"target_test"``.
    """
    source_dataset = MemmapDataset(
        data_dir=data_dir,
        split="all",
        envs=list(source_envs),
        seed=seed,
        build_targets=False,
    )
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=memmap_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    target_loaders: dict[str, DataLoader] = {}
    for split in SPLIT_NAMES:  # ("train", "val", "test")
        # NOTE: MemmapDataset._build_split treats "test" identically to
        # "val" (both return val_indices).  target_test and target_val
        # currently reference the same data subset.
        dataset = MemmapDataset(
            data_dir=data_dir,
            split=split,
            envs=list(target_envs),
            seed=seed,
            build_targets=False,
        )
        should_shuffle = split == "train"
        target_loaders[f"target_{split}"] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            num_workers=num_workers,
            collate_fn=memmap_collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    return {
        "source_train": source_loader,
        **target_loaders,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NPY memmap dataloader preview")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to the NPY memmap dataset directory")
    parser.add_argument("--preview", action="store_true", help="Load one sample from each split and print its shapes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.dataset_root)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    for split in SPLIT_NAMES:
        dataset = MemmapDataset(data_dir=data_dir, split=split, build_targets=False)
        print(f"{split}: {len(dataset)} samples")

    if args.preview:
        for split in SPLIT_NAMES:
            dataset = MemmapDataset(data_dir=data_dir, split=split, build_targets=False)
            sample = dataset[0]
            print(f"{split}_preview: csi={tuple(sample['csi'].shape)}, kpts18={tuple(sample['kpts18'].shape)}, meta={sample['meta']}")


if __name__ == "__main__":
    main()