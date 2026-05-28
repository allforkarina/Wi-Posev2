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
        heatmap_size: int = 36,
        heatmap_sigma: float = 1.5,
        paf_width: float = 1.0,
        pose_range: tuple[float, float] = (-0.8, 0.8),
        few_shot_frames: int = 0,
        few_shot_subjects: int = 0,
    ) -> None:
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"split must be train/val/test/all, got {split}")
        self.split = split
        self.normalize = normalize
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.paf_width = paf_width
        self.pose_range = pose_range

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
        self.indices = self._sample_few_shot(
            self.indices, few_shot_frames, few_shot_subjects,
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

        if split == "all":
            return np.asarray(sorted(candidate_indices), dtype=np.int64)

        # Subject-level split: group by subject, shuffle subjects, 8:2
        rng = random.Random(seed)
        grouped: dict[str, list[int]] = {}
        for idx in candidate_indices:
            grouped.setdefault(sample_list[idx], []).append(idx)

        subject_ids = sorted(grouped.keys())
        rng.shuffle(subject_ids)
        n_val = max(1, int(len(subject_ids) * random_val_ratio))
        val_subjects = set(subject_ids[:n_val])
        train_subjects_set = set(subject_ids[n_val:])

        train_indices: list[int] = []
        val_indices: list[int] = []
        for subject, indices in sorted(grouped.items()):
            if subject in train_subjects_set:
                train_indices.extend(indices)
            else:
                val_indices.extend(indices)

        if split == "train":
            return np.asarray(sorted(train_indices), dtype=np.int64)
        else:
            return np.asarray(sorted(val_indices), dtype=np.int64)

    def _sample_few_shot(
        self,
        indices: np.ndarray,
        few_shot_frames: int,
        few_shot_subjects: int,
    ) -> np.ndarray:
        """Deterministic few-shot sampling.

        Subjects: sorted by ID, take first ``few_shot_subjects``.
        Frames: per (action, subject) group, sort frames by index,
        then uniformly sample ``few_shot_frames`` frames via linspace.
        """
        if few_shot_frames <= 0 and few_shot_subjects <= 0:
            return indices

        grouped: dict[tuple[str, str], list[int]] = {}
        for idx in indices:
            idx_int = int(idx)
            key = (str(self._actions[idx_int]), str(self._samples[idx_int]))
            grouped.setdefault(key, []).append(idx_int)

        if few_shot_subjects > 0:
            all_subjects = sorted(set(k[1] for k in grouped))
            chosen_subjects = set(all_subjects[:few_shot_subjects])
            grouped = {k: v for k, v in grouped.items() if k[1] in chosen_subjects}

        result: list[int] = []
        for (_action, _subject), frame_indices in sorted(grouped.items()):
            if few_shot_frames > 0:
                sorted_frames = sorted(frame_indices)
                n = min(few_shot_frames, len(sorted_frames))
                linspace_indices = np.linspace(0, len(sorted_frames) - 1, n, dtype=int)
                sampled = [sorted_frames[i] for i in linspace_indices]
            else:
                sampled = frame_indices
            result.extend(sampled)

        return np.asarray(sorted(result), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        frame_idx = int(self.indices[index])

        csi = np.array(self._csi[frame_idx])
        kpts18 = self._kpts18[frame_idx].copy()

        return {
            "csi": torch.from_numpy(csi),
            "kpts18": torch.from_numpy(np.ascontiguousarray(kpts18)),
            "meta": {
                "env": str(self._envs[frame_idx]),
                "subject": str(self._samples[frame_idx]),
                "action": str(self._actions[frame_idx]),
                "frame_idx": int(frame_idx),
            },
        }