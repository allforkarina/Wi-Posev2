from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


CSI_FILES = {
    "global_minmax": "csi_gminmax.npy",
    "global_zscore": "csi_gzscore.npy",
    "zscore": "csi_zscore.npy",
}


class MemmapDataset(Dataset):
    """Memory-mapped .npy dataset for fast training I/O.

    CSI is stored as 3 pre-normalized .npy files, read via np.load(mmap_mode='r').
    Keypoints and meta are small enough to load entirely into RAM at init.

    No HDF5 overhead, no compression — OS page cache handles I/O.
    Multiple DataLoader workers share the same OS buffer cache (mmap MAP_SHARED).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        envs: Iterable[str] | None = None,
        train_subjects: Iterable[str] | None = None,
        test_subjects: Iterable[str] | None = None,
        random_val_ratio: float = 0.2,
        seed: int = 42,
        normalize: str = "global_minmax",
    ) -> None:
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"split must be train/val/test/all, got {split}")
        self.split = split
        self.normalize = normalize

        data_dir = Path(data_dir)

        if normalize not in CSI_FILES:
            raise ValueError(f"Unknown normalize mode: {normalize}, expected one of {list(CSI_FILES)}")

        self._csi = np.load(str(data_dir / CSI_FILES[normalize]), mmap_mode="r")

        self._kpts18 = np.load(str(data_dir / "ground_truth.npy"))

        meta = np.load(str(data_dir / "meta.npz"), allow_pickle=True)
        self._envs = meta["environment"]
        self._samples = meta["sample"]
        self._actions = meta["action"]

        self.indices = self._build_split(
            split, envs, train_subjects, test_subjects, random_val_ratio, seed
        )

    def _build_split(
        self,
        split: str,
        envs: Iterable[str] | None,
        train_subjects: Iterable[str] | None,
        test_subjects: Iterable[str] | None,
        random_val_ratio: float,
        seed: int,
    ) -> np.ndarray:
        num_total = len(self._samples)
        env_list = [str(e) for e in self._envs]
        sample_list = [str(s) for s in self._samples]

        env_set = set(envs) if envs else None
        subject_set = set(train_subjects) if train_subjects else None

        candidate_indices: list[int] = []
        for i in range(num_total):
            if env_set is not None and env_list[i] not in env_set:
                continue
            if subject_set is not None and sample_list[i] not in subject_set:
                continue
            candidate_indices.append(i)

        if split != "all":
            rng = random.Random(seed)
            grouped: dict[str, list[int]] = {}
            for idx in candidate_indices:
                grouped.setdefault(sample_list[idx], []).append(idx)

            train_indices: list[int] = []
            val_indices: list[int] = []
            for subject, indices in sorted(grouped.items()):
                shuffled = indices[:]
                rng.shuffle(shuffled)
                pivot = int(round(len(shuffled) * (1.0 - random_val_ratio)))
                train_indices.extend(shuffled[:pivot])
                val_indices.extend(shuffled[pivot:])

            if split == "train":
                return np.asarray(sorted(train_indices), dtype=np.int64)
            else:
                return np.asarray(sorted(val_indices), dtype=np.int64)

        return np.asarray(sorted(candidate_indices), dtype=np.int64)

    def _sample_few_shot(
        self,
        few_shot_subjects: int,
        few_shot_frames: int,
    ) -> list[int]:
        action_list = [str(a) for a in self._actions]
        sample_list = [str(s) for s in self._samples]

        unique_subjects = sorted(set(sample_list))
        selected_subjects = unique_subjects[:few_shot_subjects]

        selected_indices: list[int] = []
        for subject in selected_subjects:
            for action in sorted(set(action_list)):
                group = [
                    i for i in range(len(self._actions))
                    if sample_list[i] == subject and action_list[i] == action
                ]
                if not group:
                    continue
                group.sort()
                if len(group) <= few_shot_frames:
                    selected_indices.extend(group)
                else:
                    sampled = np.linspace(0, len(group) - 1, few_shot_frames, dtype=int)
                    selected_indices.extend([group[s] for s in sampled])

        return sorted(selected_indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        frame_idx = int(self.indices[index])

        csi = np.array(self._csi[frame_idx])
        kpts18 = self._kpts18[frame_idx].copy()

        item: dict = {
            "csi": torch.from_numpy(csi),
            "kpts18": torch.from_numpy(np.ascontiguousarray(kpts18)),
            "meta": {
                "env": str(self._envs[frame_idx]),
                "subject": str(self._samples[frame_idx]),
                "action": str(self._actions[frame_idx]),
                "frame_idx": int(frame_idx),
            },
        }
        return item