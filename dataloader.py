from __future__ import annotations

"""NPY memmap-backed dataloader for MM-Fi pose data."""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.memmap_dataset import MemmapDataset

ALL_SPLITS = ("train", "val", "test", "all")


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
    envs: Sequence[str] | None = None,
) -> DataLoader:
    if split not in ALL_SPLITS:
        raise ValueError(f"split must be one of {ALL_SPLITS}, got {split}")

    dataset = MemmapDataset(
        data_dir=data_dir,
        split=split,
        seed=seed,
        envs=list(envs) if envs else None,
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
    for split in ("train", "val", "test"):
        dataset = MemmapDataset(
            data_dir=data_dir,
            split=split,
            envs=list(target_envs),
            seed=seed,
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


def create_few_shot_data_loader(
    data_dir: str | Path,
    envs: Sequence[str],
    batch_size: int,
    few_shot_frames: int = 5,
    few_shot_subjects: int = 4,
    num_workers: int = 0,
    seed: int = 42,
) -> dict:
    """Create train/val DataLoaders for few-shot fine-tuning.

    Train: few-shot sampled subset of *envs* (≤ few_shot_subjects subjects,
    ≤ few_shot_frames per action×subject via deterministic uniform spacing).

    Val: ALL data in *envs* excluding the few-shot train indices.

    Returns dict with keys: ``"train"``, ``"val"``, ``"train_indices"``.
    """
    train_dataset = MemmapDataset(
        data_dir=data_dir, split="all", envs=list(envs),
        seed=seed,
        few_shot_frames=few_shot_frames,
        few_shot_subjects=few_shot_subjects,
    )
    train_indices = train_dataset.indices.copy()

    # Val = full target domain minus few-shot train
    full_dataset = MemmapDataset(
        data_dir=data_dir, split="all", envs=list(envs),
        seed=seed,
    )
    train_idx_set = set(int(i) for i in train_indices)
    val_indices = np.asarray(sorted(
        [i for i in full_dataset.indices if int(i) not in train_idx_set]
    ), dtype=np.int64)

    val_dataset = MemmapDataset(
        data_dir=data_dir, split="all", envs=list(envs),
        seed=seed,
    )
    val_dataset.indices = val_indices

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=memmap_collate_fn,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=memmap_collate_fn,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
    return {"train": train_loader, "val": val_loader, "train_indices": train_indices}
