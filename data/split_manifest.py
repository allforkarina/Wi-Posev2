from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ENVIRONMENTS = ("env1", "env2")
SPLITS = ("train", "val", "test")
SPLIT_MODES = ("random_frame", "temporal_block")
DEFAULT_FEW_SHOT_SPECS: dict[str, int] = {
    "env2_fewshot_540": 2,
    "env2_fewshot_810": 3,
    "env2_fewshot_4050": 15,
    "env2_fewshot_8100": 30,
}


@dataclass(frozen=True)
class DatasetMetadata:
    environment: np.ndarray
    subject: np.ndarray
    action: np.ndarray

    def __post_init__(self) -> None:
        lengths = {len(self.environment), len(self.subject), len(self.action)}
        if len(lengths) != 1:
            raise ValueError("Metadata environment, subject, and action arrays must have equal lengths")

    @classmethod
    def from_npz(cls, path: str | Path) -> DatasetMetadata:
        with np.load(path, allow_pickle=True) as meta:
            return cls(
                environment=np.asarray(meta["environment"]).astype(str),
                subject=np.asarray(meta["sample"]).astype(str),
                action=np.asarray(meta["action"]).astype(str),
            )


@dataclass(frozen=True)
class SplitManifest:
    path: Path
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]
    manifest_hash: str

    @property
    def mode(self) -> str:
        return str(self.metadata["mode"])

    @property
    def source_train_normalization(self) -> tuple[float, float]:
        return (
            float(self.metadata["source_train_min"]),
            float(self.metadata["source_train_max"]),
        )

    def indices(self, key: str) -> np.ndarray:
        if key not in self.arrays:
            raise KeyError(f"Manifest has no split key: {key}")
        return np.asarray(self.arrays[key], dtype=np.int64)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.int64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def compute_source_train_normalization(
    dataset_root: str | Path,
    train_indices: np.ndarray,
    chunk_size: int = 4096,
) -> tuple[float, float]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    indices = np.asarray(train_indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("Source training indices cannot be empty")
    csi = np.load(Path(dataset_root) / "csi_gminmax.npy", mmap_mode="r")
    lower = float("inf")
    upper = float("-inf")
    for start in range(0, len(indices), chunk_size):
        chunk = np.asarray(csi[indices[start:start + chunk_size]])
        lower = min(lower, float(chunk.min()))
        upper = max(upper, float(chunk.max()))
    if upper - lower <= 1e-12:
        raise ValueError("Source-train normalization range must be greater than 1e-12")
    return lower, upper


def stable_group_seed(seed: int, group: tuple[str, str, str], purpose: str) -> int:
    payload = "\0".join((str(seed), *group, purpose)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _group_rows(metadata: DatasetMetadata) -> dict[tuple[str, str, str], np.ndarray]:
    groups: dict[tuple[str, str, str], list[int]] = {}
    for index, (environment, subject, action) in enumerate(zip(
        metadata.environment.astype(str),
        metadata.subject.astype(str),
        metadata.action.astype(str),
    )):
        if environment not in ENVIRONMENTS:
            continue
        groups.setdefault((environment, subject, action), []).append(index)
    return {
        group: np.asarray(indices, dtype=np.int64)
        for group, indices in sorted(groups.items())
    }


def _random_frame_split(
    indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shuffled = np.random.default_rng(seed).permutation(indices)
    val_count = round(0.1 * len(indices))
    test_count = round(0.1 * len(indices))
    train_count = len(indices) - val_count - test_count
    return (
        shuffled[:train_count],
        shuffled[train_count:train_count + val_count],
        shuffled[train_count + val_count:],
    )


def _temporal_block_split(
    indices: np.ndarray,
    seed: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks = [indices[start:start + block_size] for start in range(0, len(indices), block_size)]
    if len(blocks) < 3:
        raise ValueError(
            f"Each group requires at least three temporal blocks, got {len(blocks)}"
        )
    order = np.random.default_rng(seed).permutation(len(blocks))
    blocks = [blocks[int(index)] for index in order]
    val_count = max(1, round(0.1 * len(blocks)))
    test_count = max(1, round(0.1 * len(blocks)))
    train_count = len(blocks) - val_count - test_count
    if train_count < 1:
        raise ValueError("Temporal block allocation must leave at least one training block")
    return (
        np.concatenate(blocks[:train_count]),
        np.concatenate(blocks[train_count:train_count + val_count]),
        np.concatenate(blocks[train_count + val_count:]),
    )


def build_split_arrays(
    metadata: DatasetMetadata,
    mode: str,
    seed: int,
    block_size: int = 16,
    few_shot_specs: Mapping[str, int] | None = None,
) -> dict[str, np.ndarray]:
    if mode not in SPLIT_MODES:
        raise ValueError(f"mode must be one of {SPLIT_MODES}, got {mode}")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    specs = DEFAULT_FEW_SHOT_SPECS if few_shot_specs is None else dict(few_shot_specs)
    grouped = _group_rows(metadata)
    accumulators: dict[str, list[np.ndarray]] = {
        f"{environment}_{split}": []
        for environment in ENVIRONMENTS
        for split in SPLITS
    }

    for group, indices in grouped.items():
        environment = group[0]
        group_seed = stable_group_seed(seed, group, mode)
        if mode == "random_frame":
            group_splits = _random_frame_split(indices, group_seed)
        else:
            group_splits = _temporal_block_split(indices, group_seed, block_size)
        for split, values in zip(SPLITS, group_splits):
            accumulators[f"{environment}_{split}"].append(values)

    arrays = {
        key: np.sort(np.concatenate(parts)).astype(np.int64)
        if parts else np.empty(0, dtype=np.int64)
        for key, parts in accumulators.items()
    }
    _attach_few_shot_arrays(arrays, metadata, seed, specs)
    validate_manifest(arrays, metadata, tuple(specs))
    return arrays


def _attach_few_shot_arrays(
    arrays: dict[str, np.ndarray],
    metadata: DatasetMetadata,
    seed: int,
    few_shot_specs: Mapping[str, int],
) -> None:
    if not few_shot_specs:
        return
    quotas = list(few_shot_specs.values())
    if quotas != sorted(quotas) or len(set(quotas)) != len(quotas):
        raise ValueError("Few-shot frame quotas must be strictly increasing")

    env2_train = set(arrays["env2_train"].tolist())
    ordered_groups: dict[tuple[str, str, str], np.ndarray] = {}
    for group, group_indices in _group_rows(metadata).items():
        if group[0] != "env2":
            continue
        candidates = np.asarray(
            [index for index in group_indices if int(index) in env2_train],
            dtype=np.int64,
        )
        maximum = quotas[-1]
        if len(candidates) < maximum:
            raise ValueError(
                f"Group {group} requires {maximum} training frames for few-shot selection, "
                f"got {len(candidates)}"
            )
        order_seed = stable_group_seed(seed, group, "few_shot")
        ordered_groups[group] = np.random.default_rng(order_seed).permutation(candidates)

    for key, quota in few_shot_specs.items():
        arrays[key] = np.sort(np.concatenate([
            ordered[:quota]
            for ordered in ordered_groups.values()
        ])).astype(np.int64)


def validate_manifest(
    arrays: Mapping[str, np.ndarray],
    metadata: DatasetMetadata,
    few_shot_keys: tuple[str, ...] = tuple(DEFAULT_FEW_SHOT_SPECS),
) -> None:
    required = {
        f"{environment}_{split}"
        for environment in ENVIRONMENTS
        for split in SPLITS
    }
    required.update(few_shot_keys)
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"Manifest is missing keys: {missing}")

    total = len(metadata.environment)
    for key in required:
        values = np.asarray(arrays[key])
        if values.dtype != np.int64:
            raise ValueError(f"Manifest key {key} must use int64 indices")
        if len(values) != len(np.unique(values)):
            raise ValueError(f"Manifest key {key} contains duplicate indices")
        if np.any(values < 0) or np.any(values >= total):
            raise ValueError(f"Manifest key {key} contains out-of-range indices")

    environment_values = metadata.environment.astype(str)
    for environment in ENVIRONMENTS:
        split_sets = {
            split: set(np.asarray(arrays[f"{environment}_{split}"]).tolist())
            for split in SPLITS
        }
        if any(
            split_sets[left] & split_sets[right]
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise ValueError(f"{environment} train/val/test indices overlap")
        expected = set(np.flatnonzero(environment_values == environment).tolist())
        if set.union(*split_sets.values()) != expected:
            raise ValueError(
                f"{environment} train/val/test indices do not cover the environment"
            )

    env2_train = set(np.asarray(arrays["env2_train"]).tolist())
    previous: set[int] = set()
    for key in few_shot_keys:
        current = set(np.asarray(arrays[key]).tolist())
        if not current <= env2_train:
            raise ValueError(f"Manifest key {key} contains values outside env2_train")
        if not previous <= current:
            raise ValueError(f"Manifest few-shot sets are not nested at {key}")
        previous = current


def save_manifest(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    dataset_root: str | Path,
    mode: str,
    seed: int,
    block_size: int,
    source_train_normalization: tuple[float, float],
) -> None:
    path = Path(path)
    dataset_root = Path(dataset_root)
    few_shot_keys = tuple(key for key in arrays if key.startswith("env2_fewshot_"))
    metadata = DatasetMetadata.from_npz(dataset_root / "meta.npz")
    validate_manifest(arrays, metadata, few_shot_keys)
    lower, upper = source_train_normalization
    if upper - lower <= 1e-12:
        raise ValueError("Source-train normalization range must be greater than 1e-12")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    sidecar = {
        "mode": mode,
        "seed": int(seed),
        "block_size": int(block_size),
        "counts": {key: int(len(values)) for key, values in arrays.items()},
        "meta_sha256": sha256_file(dataset_root / "meta.npz"),
        "array_sha256": {key: sha256_array(values) for key, values in arrays.items()},
        "source_train_min": float(lower),
        "source_train_max": float(upper),
    }
    path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_manifest(
    path: str | Path,
    dataset_root: str | Path,
    few_shot_keys: tuple[str, ...] = tuple(DEFAULT_FEW_SHOT_SPECS),
) -> SplitManifest:
    path = Path(path)
    dataset_root = Path(dataset_root)
    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar["meta_sha256"] != sha256_file(dataset_root / "meta.npz"):
        raise ValueError("Manifest metadata fingerprint does not match dataset meta.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            key: np.asarray(archive[key], dtype=np.int64)
            for key in archive.files
        }
    for key, expected_hash in sidecar["array_sha256"].items():
        if key not in arrays or sha256_array(arrays[key]) != expected_hash:
            raise ValueError(f"Manifest array hash mismatch for key: {key}")
    metadata = DatasetMetadata.from_npz(dataset_root / "meta.npz")
    validate_manifest(arrays, metadata, few_shot_keys)
    return SplitManifest(
        path=path.resolve(),
        arrays=arrays,
        metadata=sidecar,
        manifest_hash=sha256_file(path),
    )
