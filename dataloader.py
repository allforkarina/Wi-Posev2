from __future__ import annotations

"""NPY memmap-backed dataloader for MM-Fi pose data."""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

from data.memmap_dataset import MemmapDataset

SPLIT_NAMES = ("train", "val", "test")
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
    envs: tuple[str, ...] | None = None,
    num_workers: int = 0,
    shuffle: bool | None = None,
    seed: int = 42,
) -> DataLoader:
    if split not in ALL_SPLITS:
        raise ValueError(f"split must be one of {ALL_SPLITS}, got {split}")

    dataset = MemmapDataset(
        data_dir=data_dir,
        split=split,
        envs=envs,
        seed=seed,
    )
    should_shuffle = shuffle if shuffle is not None else split in ("train", "all")
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


def create_few_shot_data_loader(
    data_dir: str | Path,
    target_envs: tuple[str, ...],
    few_shot_subjects: int,
    few_shot_frames: int,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[int]]:
    full_dataset = MemmapDataset(
        data_dir=data_dir,
        split="all",
        envs=target_envs,
        seed=seed,
    )
    train_indices = full_dataset._sample_few_shot(
        few_shot_subjects=few_shot_subjects,
        few_shot_frames=few_shot_frames,
    )
    train_dataset = Subset(full_dataset, train_indices)
    all_indices = list(range(len(full_dataset)))
    val_indices = [i for i in all_indices if i not in set(train_indices)]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=memmap_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_dataset = Subset(full_dataset, val_indices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=memmap_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_indices