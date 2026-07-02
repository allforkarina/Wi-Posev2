from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.split_manifest import (
    DatasetMetadata,
    SplitManifest,
    build_split_arrays,
    load_manifest,
    save_manifest,
    validate_manifest,
)
from scripts.build_split_manifests import main as build_manifests_main


TEST_FEW_SHOT_SPECS = {
    "env2_fewshot_8": 2,
    "env2_fewshot_12": 3,
    "env2_fewshot_60": 15,
    "env2_fewshot_120": 30,
}


def _synthetic_metadata(frames_per_group: int) -> DatasetMetadata:
    environments: list[str] = []
    subjects: list[str] = []
    actions: list[str] = []
    for environment in ("env1", "env2"):
        for subject in ("S01", "S02"):
            for action in ("A01", "A02"):
                environments.extend([environment] * frames_per_group)
                subjects.extend([subject] * frames_per_group)
                actions.extend([action] * frames_per_group)
    return DatasetMetadata(
        environment=np.asarray(environments),
        subject=np.asarray(subjects),
        action=np.asarray(actions),
    )


def _group_indices(
    metadata: DatasetMetadata,
    group: tuple[str, str, str],
) -> np.ndarray:
    environment, subject, action = group
    return np.flatnonzero(
        (metadata.environment.astype(str) == environment)
        & (metadata.subject.astype(str) == subject)
        & (metadata.action.astype(str) == action)
    )


def _write_meta(path: Path, metadata: DatasetMetadata) -> None:
    np.savez(
        path,
        environment=metadata.environment,
        sample=metadata.subject,
        action=metadata.action,
    )


def test_random_frame_split_is_stratified_complete_and_deterministic() -> None:
    metadata = _synthetic_metadata(frames_per_group=10)
    first = build_split_arrays(
        metadata,
        mode="random_frame",
        seed=42,
        block_size=16,
        few_shot_specs={},
    )
    second = build_split_arrays(
        metadata,
        mode="random_frame",
        seed=42,
        block_size=16,
        few_shot_specs={},
    )

    for environment in ("env1", "env2"):
        assert len(first[f"{environment}_train"]) == 32
        assert len(first[f"{environment}_val"]) == 4
        assert len(first[f"{environment}_test"]) == 4
        combined = np.concatenate([
            first[f"{environment}_train"],
            first[f"{environment}_val"],
            first[f"{environment}_test"],
        ])
        assert len(np.unique(combined)) == 40

    assert first.keys() == second.keys()
    assert all(np.array_equal(first[key], second[key]) for key in first)


def test_random_frame_split_uses_stable_independent_group_ordering() -> None:
    metadata = _synthetic_metadata(frames_per_group=10)
    arrays = build_split_arrays(
        metadata,
        mode="random_frame",
        seed=42,
        block_size=16,
        few_shot_specs={},
    )

    for environment in ("env1", "env2"):
        for subject in ("S01", "S02"):
            for action in ("A01", "A02"):
                group = set(_group_indices(metadata, (environment, subject, action)).tolist())
                assert len(group & set(arrays[f"{environment}_train"].tolist())) == 8
                assert len(group & set(arrays[f"{environment}_val"].tolist())) == 1
                assert len(group & set(arrays[f"{environment}_test"].tolist())) == 1


def test_temporal_blocks_never_cross_split_boundaries() -> None:
    metadata = _synthetic_metadata(frames_per_group=160)
    arrays = build_split_arrays(
        metadata,
        mode="temporal_block",
        seed=42,
        block_size=16,
        few_shot_specs={},
    )

    for environment in ("env1", "env2"):
        ownership = {
            int(index): split
            for split in ("train", "val", "test")
            for index in arrays[f"{environment}_{split}"]
        }
        group = _group_indices(metadata, (environment, "S01", "A01"))
        for start in range(0, len(group), 16):
            assert len({ownership[int(i)] for i in group[start:start + 16]}) == 1


def test_temporal_split_rejects_group_with_fewer_than_three_blocks() -> None:
    metadata = _synthetic_metadata(frames_per_group=32)
    with pytest.raises(ValueError, match="at least three temporal blocks"):
        build_split_arrays(
            metadata,
            mode="temporal_block",
            seed=42,
            block_size=16,
            few_shot_specs={},
        )


def test_few_shot_sets_are_balanced_nested_and_train_only() -> None:
    metadata = _synthetic_metadata(frames_per_group=64)
    arrays = build_split_arrays(
        metadata,
        mode="random_frame",
        seed=42,
        block_size=16,
        few_shot_specs=TEST_FEW_SHOT_SPECS,
    )

    assert [len(arrays[key]) for key in TEST_FEW_SHOT_SPECS] == [8, 12, 60, 120]
    keys = list(TEST_FEW_SHOT_SPECS)
    for smaller, larger in zip(keys, keys[1:]):
        assert set(arrays[smaller].tolist()) < set(arrays[larger].tolist())
    assert set(arrays[keys[-1]].tolist()) <= set(arrays["env2_train"].tolist())
    assert not set(arrays[keys[-1]].tolist()) & set(arrays["env2_val"].tolist())
    assert not set(arrays[keys[-1]].tolist()) & set(arrays["env2_test"].tolist())

    for key, quota in TEST_FEW_SHOT_SPECS.items():
        chosen = set(arrays[key].tolist())
        for subject in ("S01", "S02"):
            for action in ("A01", "A02"):
                group = set(_group_indices(metadata, ("env2", subject, action)).tolist())
                assert len(chosen & group) == quota


def test_few_shot_rejects_insufficient_training_frames() -> None:
    metadata = _synthetic_metadata(frames_per_group=20)
    with pytest.raises(ValueError, match="requires 30 training frames"):
        build_split_arrays(
            metadata,
            mode="random_frame",
            seed=42,
            block_size=16,
            few_shot_specs=TEST_FEW_SHOT_SPECS,
        )


def test_validation_rejects_overlap_and_non_nested_few_shot() -> None:
    metadata = _synthetic_metadata(frames_per_group=64)
    arrays = build_split_arrays(
        metadata,
        mode="random_frame",
        seed=42,
        block_size=16,
        few_shot_specs=TEST_FEW_SHOT_SPECS,
    )
    arrays["env1_val"] = np.append(
        arrays["env1_val"], arrays["env1_train"][0]
    ).astype(np.int64)
    with pytest.raises(ValueError, match="env1 train/val/test indices overlap"):
        validate_manifest(arrays, metadata, tuple(TEST_FEW_SHOT_SPECS))

    arrays = build_split_arrays(
        metadata,
        mode="random_frame",
        seed=42,
        block_size=16,
        few_shot_specs=TEST_FEW_SHOT_SPECS,
    )
    arrays["env2_fewshot_12"] = arrays["env2_fewshot_12"][1:]
    with pytest.raises(ValueError, match="not nested"):
        validate_manifest(arrays, metadata, tuple(TEST_FEW_SHOT_SPECS))


def test_saved_manifest_reloads_and_rejects_metadata_mismatch(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    metadata = _synthetic_metadata(frames_per_group=64)
    _write_meta(dataset_root / "meta.npz", metadata)
    arrays = build_split_arrays(
        metadata,
        mode="random_frame",
        seed=42,
        block_size=16,
        few_shot_specs=TEST_FEW_SHOT_SPECS,
    )
    path = tmp_path / "random_frame_seed42.npz"

    save_manifest(
        path,
        arrays,
        dataset_root=dataset_root,
        mode="random_frame",
        seed=42,
        block_size=16,
        source_train_normalization=(0.1, 0.9),
    )
    manifest = load_manifest(path, dataset_root, few_shot_keys=tuple(TEST_FEW_SHOT_SPECS))

    assert isinstance(manifest, SplitManifest)
    assert manifest.mode == "random_frame"
    assert manifest.source_train_normalization == (0.1, 0.9)
    assert np.array_equal(manifest.indices("env1_train"), arrays["env1_train"])
    assert len(manifest.manifest_hash) == 64

    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert set(sidecar["array_sha256"]) == set(arrays)

    changed = _synthetic_metadata(frames_per_group=65)
    _write_meta(dataset_root / "meta.npz", changed)
    with pytest.raises(ValueError, match="metadata fingerprint"):
        load_manifest(path, dataset_root, few_shot_keys=tuple(TEST_FEW_SHOT_SPECS))


def test_load_manifest_rejects_tampered_array(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    metadata = _synthetic_metadata(frames_per_group=64)
    _write_meta(dataset_root / "meta.npz", metadata)
    arrays = build_split_arrays(
        metadata,
        mode="random_frame",
        seed=42,
        block_size=16,
        few_shot_specs=TEST_FEW_SHOT_SPECS,
    )
    path = tmp_path / "random_frame_seed42.npz"
    save_manifest(
        path,
        arrays,
        dataset_root=dataset_root,
        mode="random_frame",
        seed=42,
        block_size=16,
        source_train_normalization=(0.1, 0.9),
    )
    arrays["env1_train"] = arrays["env1_train"].copy()
    arrays["env1_train"][0] += 1
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match="array hash mismatch"):
        load_manifest(path, dataset_root, few_shot_keys=tuple(TEST_FEW_SHOT_SPECS))


def test_builder_cli_creates_reloadable_protocol_manifests(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "manifests"
    dataset_root.mkdir()
    metadata = _synthetic_metadata(frames_per_group=64)
    _write_meta(dataset_root / "meta.npz", metadata)
    values = np.linspace(0.0, 1.0, len(metadata.environment), dtype=np.float32)
    csi = np.broadcast_to(values[:, None, None, None], (len(values), 2, 3, 4)).copy()
    np.save(dataset_root / "csi_gminmax.npy", csi)

    exit_code = build_manifests_main([
        "--dataset-root", str(dataset_root),
        "--output-dir", str(output_dir),
        "--seed", "42",
        "--block-size", "16",
    ])

    assert exit_code == 0
    random_path = output_dir / "random_frame_seed42.npz"
    temporal_path = output_dir / "temporal_block16_seed42.npz"
    assert random_path.is_file()
    assert random_path.with_suffix(".json").is_file()
    assert temporal_path.is_file()
    assert temporal_path.with_suffix(".json").is_file()
    random_manifest = load_manifest(random_path, dataset_root)
    temporal_manifest = load_manifest(temporal_path, dataset_root)
    assert random_manifest.mode == "random_frame"
    assert temporal_manifest.mode == "temporal_block"
