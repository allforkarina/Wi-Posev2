"""Build the delivery memmap directly from raw CSI and audited GT arrays.

The builder preserves the first two GT channels exactly before applying the
project's verified 17-to-18 mapping.  It never interprets ``(0, 0)`` as
missing, never assumes an image resolution, and never clips pose coordinates.

CSI is staged in a disk-backed raw memmap and normalized in chunks, keeping
peak RAM bounded.  ``--resume`` continues after the last completed trial or
normalization chunk following a safe interruption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import scipy.io as sio
from scipy.signal import resample

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pose_schema import (  # noqa: E402
    CANONICAL_BONE_EDGES,
    JOINT_NAMES,
    MAPPED_18_TO_RAW_17,
    SYNTHETIC_JOINT_INDEX,
    SYNTHETIC_JOINT_SOURCES,
    map_raw17_to_project18,
)


TIME_PACKETS = 64
RX_ANTENNAS = 3
SUBCARRIERS = 114
STATE_FILENAME = "build_state.json"
COMPLETE_FILENAME = "build_complete.json"
RAW_CSI_FILENAME = "csi_raw.build.npy"


@dataclass(frozen=True)
class TrialSpec:
    path: Path
    action: str
    subject: str
    environment: str
    frame_paths: tuple[Path, ...]
    frame_indices: tuple[int, ...]
    gt_path: Path | None
    rgb_paths: tuple[Path, ...]

    @property
    def frame_count(self) -> int:
        return len(self.frame_paths)


def derive_env(subject: str) -> str:
    number = int(subject.removeprefix("S"))
    return f"env{(number - 1) // 10 + 1}"


def sanitize_csi(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(result)
    if finite.all():
        return result
    fill = float(np.median(result[finite])) if finite.any() else 0.0
    return np.nan_to_num(result, nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def preprocess_csi_one_frame(csi_amplitude: np.ndarray) -> np.ndarray:
    values = sanitize_csi(csi_amplitude)
    values = sanitize_csi(resample(values, TIME_PACKETS, axis=-1))
    expected = (RX_ANTENNAS, SUBCARRIERS, TIME_PACKETS)
    if values.shape != expected:
        raise ValueError(f"Expected resampled CSI shaped {expected}, got {values.shape}")
    return np.transpose(values, (2, 0, 1)).astype(np.float32, copy=False)


def _frame_number(path: Path) -> int:
    suffix = path.stem.removeprefix("frame")
    if not suffix.isdigit():
        raise ValueError(f"Unexpected frame filename: {path}")
    return int(suffix)


def _gt_filename(environment: str, subject: str, action: str) -> str:
    env_number = int(environment.removeprefix("env"))
    return f"E{env_number:02d}_{subject}_{action}.npy"


def scan_trials(src_root: Path, gt_dir: Path | None) -> list[TrialSpec]:
    if not src_root.is_dir():
        raise FileNotFoundError(f"Raw dataset root not found: {src_root}")
    if gt_dir is not None and not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")

    specs: list[TrialSpec] = []
    for action_dir in sorted(src_root.glob("A*")):
        if not action_dir.is_dir():
            continue
        for subject_dir in sorted(action_dir.glob("S*")):
            if not subject_dir.is_dir():
                continue
            wifi_paths = tuple(sorted((subject_dir / "wifi-csi").glob("frame*.mat")))
            if not wifi_paths:
                continue
            environment = derive_env(subject_dir.name)
            if gt_dir is not None:
                gt_path = gt_dir / _gt_filename(
                    environment,
                    subject_dir.name,
                    action_dir.name,
                )
                if not gt_path.is_file():
                    raise FileNotFoundError(f"Missing GT file: {gt_path}")
                gt = np.load(gt_path, mmap_mode="r")
                if gt.ndim != 3 or gt.shape[1] != 17 or gt.shape[2] < 2:
                    raise ValueError(
                        f"Expected GT shaped [N, 17, >=2], got {gt.shape}: {gt_path}"
                    )
                if len(gt) != len(wifi_paths):
                    raise ValueError(
                        f"CSI/GT frame mismatch for {gt_path.name}: "
                        f"{len(wifi_paths)} != {len(gt)}"
                    )
                selected_wifi = wifi_paths
                rgb_paths: tuple[Path, ...] = ()
            else:
                rgb_by_stem = {
                    path.stem: path
                    for path in (subject_dir / "rgb").glob("frame*.npy")
                }
                selected_wifi = tuple(path for path in wifi_paths if path.stem in rgb_by_stem)
                if not selected_wifi:
                    raise FileNotFoundError(f"No CSI/RGB frame pairs in {subject_dir}")
                rgb_paths = tuple(rgb_by_stem[path.stem] for path in selected_wifi)
                gt_path = None
            specs.append(TrialSpec(
                path=subject_dir,
                action=action_dir.name,
                subject=subject_dir.name,
                environment=environment,
                frame_paths=selected_wifi,
                frame_indices=tuple(_frame_number(path) for path in selected_wifi),
                gt_path=gt_path,
                rgb_paths=rgb_paths,
            ))
    if not specs:
        raise RuntimeError("No valid trials were found")
    return specs


def process_trial(spec: TrialSpec) -> tuple[np.ndarray, np.ndarray]:
    csi = np.empty(
        (spec.frame_count, TIME_PACKETS, RX_ANTENNAS, SUBCARRIERS),
        dtype=np.float32,
    )
    for index, path in enumerate(spec.frame_paths):
        payload = sio.loadmat(path)
        if "CSIamp" not in payload:
            raise KeyError(f"CSIamp missing from {path}")
        csi[index] = preprocess_csi_one_frame(payload["CSIamp"])

    if spec.gt_path is not None:
        raw_gt = np.asarray(np.load(spec.gt_path, mmap_mode="r")[..., :2], dtype=np.float32)
    else:
        raw_gt = np.stack([
            np.asarray(np.load(path), dtype=np.float32)[..., :2]
            for path in spec.rgb_paths
        ])
    if raw_gt.shape != (spec.frame_count, 17, 2):
        raise ValueError(
            f"Expected raw GT shaped {(spec.frame_count, 17, 2)}, got {raw_gt.shape}"
        )
    if not np.isfinite(raw_gt).all():
        raise ValueError(f"Non-finite GT coordinates found in {spec.path}")
    return csi, map_raw17_to_project18(raw_gt).astype(np.float32, copy=False)


def _iter_processed_trials(
    specs: Sequence[TrialSpec],
    start_index: int,
    workers: int,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    if workers <= 1:
        for index in range(start_index, len(specs)):
            csi, gt = process_trial(specs[index])
            yield index, csi, gt
        return

    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending: dict[int, Future[tuple[np.ndarray, np.ndarray]]] = {}
        submit_index = start_index
        window = max(workers * 2, 1)
        while submit_index < min(len(specs), start_index + window):
            pending[submit_index] = pool.submit(process_trial, specs[submit_index])
            submit_index += 1
        next_index = start_index
        while next_index < len(specs):
            csi, gt = pending.pop(next_index).result()
            yield next_index, csi, gt
            if submit_index < len(specs):
                pending[submit_index] = pool.submit(process_trial, specs[submit_index])
                submit_index += 1
            next_index += 1


def _spec_fingerprint(specs: Sequence[TrialSpec]) -> str:
    digest = hashlib.sha256()
    for spec in specs:
        digest.update(
            (
                f"{spec.environment}\0{spec.subject}\0{spec.action}\0"
                f"{spec.frame_count}\0{spec.gt_path}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _initial_state(specs: Sequence[TrialSpec]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "spec_fingerprint": _spec_fingerprint(specs),
        "phase": "staging",
        "next_trial_index": 0,
        "frame_offset": 0,
        "train_value_count": 0,
        "train_sum": 0.0,
        "train_sum_squares": 0.0,
        "train_min": None,
        "train_max": None,
        "normalization_offsets": {
            "global_zscore": 0,
            "zscore": 0,
            "global_minmax": 0,
        },
    }


def _load_or_create_state(
    dst_root: Path,
    specs: Sequence[TrialSpec],
    total_frames: int,
    resume: bool,
) -> tuple[dict[str, object], np.memmap, np.memmap]:
    state_path = dst_root / STATE_FILENAME
    raw_path = dst_root / RAW_CSI_FILENAME
    gt_path = dst_root / "ground_truth.npy"
    expected_fingerprint = _spec_fingerprint(specs)
    if state_path.is_file():
        if not resume:
            raise FileExistsError(
                f"Partial build exists at {dst_root}; rerun with --resume"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("spec_fingerprint") != expected_fingerprint:
            raise ValueError("Raw dataset/GT trial fingerprint changed since the partial build")
        raw = np.lib.format.open_memmap(raw_path, mode="r+")
        gt = np.lib.format.open_memmap(gt_path, mode="r+")
        return state, raw, gt

    conflicting = [
        dst_root / RAW_CSI_FILENAME,
        dst_root / "csi_gminmax.npy",
        dst_root / "ground_truth.npy",
        dst_root / COMPLETE_FILENAME,
    ]
    if any(path.exists() for path in conflicting):
        raise FileExistsError(
            f"Destination already contains dataset artifacts: {dst_root}"
        )
    state = _initial_state(specs)
    raw = np.lib.format.open_memmap(
        raw_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_frames, TIME_PACKETS, RX_ANTENNAS, SUBCARRIERS),
    )
    gt = np.lib.format.open_memmap(
        gt_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_frames, 18, 2),
    )
    _write_json_atomic(state_path, state)
    return state, raw, gt


def _stage_trials(
    dst_root: Path,
    specs: Sequence[TrialSpec],
    train_subjects: set[str],
    workers: int,
    state: dict[str, object],
    raw_memmap: np.memmap,
    gt_memmap: np.memmap,
) -> None:
    if state["phase"] != "staging":
        return
    start_index = int(state["next_trial_index"])
    offset = int(state["frame_offset"])
    state_path = dst_root / STATE_FILENAME
    started = time.time()
    for index, csi, gt in _iter_processed_trials(specs, start_index, workers):
        spec = specs[index]
        end = offset + spec.frame_count
        raw_memmap[offset:end] = csi
        gt_memmap[offset:end] = gt
        raw_memmap.flush()
        gt_memmap.flush()
        if spec.subject in train_subjects:
            values = csi
            count = int(values.size)
            current_min = float(values.min())
            current_max = float(values.max())
            state["train_value_count"] = int(state["train_value_count"]) + count
            state["train_sum"] = float(state["train_sum"]) + float(
                np.sum(values, dtype=np.float64)
            )
            state["train_sum_squares"] = float(state["train_sum_squares"]) + float(
                np.sum(np.square(values, dtype=np.float64), dtype=np.float64)
            )
            previous_min = state["train_min"]
            previous_max = state["train_max"]
            state["train_min"] = (
                current_min if previous_min is None else min(float(previous_min), current_min)
            )
            state["train_max"] = (
                current_max if previous_max is None else max(float(previous_max), current_max)
            )
        offset = end
        state["next_trial_index"] = index + 1
        state["frame_offset"] = offset
        _write_json_atomic(state_path, state)
        print(
            f"[{index + 1}/{len(specs)}] {spec.action}/{spec.subject} "
            f"frames={spec.frame_count} total={offset}",
            flush=True,
        )
    state["phase"] = "normalizing"
    _write_json_atomic(state_path, state)
    print(f"CSI/GT staging completed in {time.time() - started:.1f}s", flush=True)


def _normalization_stats(state: dict[str, object]) -> tuple[float, float, float, float]:
    count = int(state["train_value_count"])
    if count < 1 or state["train_min"] is None or state["train_max"] is None:
        raise ValueError("No source-training CSI values were accumulated")
    mean = float(state["train_sum"]) / count
    variance = max(float(state["train_sum_squares"]) / count - mean * mean, 0.0)
    std = variance ** 0.5
    lower = float(state["train_min"])
    upper = float(state["train_max"])
    if upper - lower <= 1e-12 or std <= 1e-12:
        raise ValueError("Degenerate source-training CSI normalization statistics")
    return lower, upper, mean, std


def _normalize_variant(
    dst_root: Path,
    raw: np.memmap,
    state: dict[str, object],
    variant: str,
    chunk_size: int,
    lower: float,
    upper: float,
    mean: float,
    std: float,
) -> None:
    output_name = {
        "global_minmax": "csi_gminmax.npy",
        "global_zscore": "csi_gzscore.npy",
        "zscore": "csi_zscore.npy",
    }[variant]
    output_path = dst_root / output_name
    offsets = dict(state["normalization_offsets"])
    start_offset = int(offsets[variant])
    mode = "r+" if output_path.is_file() else "w+"
    output = np.lib.format.open_memmap(
        output_path,
        mode=mode,
        dtype=np.float32,
        shape=raw.shape,
    )
    state_path = dst_root / STATE_FILENAME
    for start in range(start_offset, len(raw), chunk_size):
        end = min(start + chunk_size, len(raw))
        values = np.asarray(raw[start:end], dtype=np.float32)
        if variant == "global_minmax":
            normalized = (values - lower) / (upper - lower)
        elif variant == "global_zscore":
            normalized = (values - mean) / std
        else:
            sample_mean = values.mean(axis=(1, 2, 3), keepdims=True)
            sample_std = values.std(axis=(1, 2, 3), keepdims=True)
            normalized = (values - sample_mean) / np.maximum(sample_std, 1e-6)
        output[start:end] = normalized.astype(np.float32, copy=False)
        output.flush()
        offsets[variant] = end
        state["normalization_offsets"] = offsets
        _write_json_atomic(state_path, state)
        print(f"normalize {variant}: {end}/{len(raw)}", flush=True)


def _metadata_arrays(specs: Sequence[TrialSpec]) -> dict[str, np.ndarray]:
    environments: list[str] = []
    subjects: list[str] = []
    actions: list[str] = []
    frame_indices: list[int] = []
    for spec in specs:
        environments.extend([spec.environment] * spec.frame_count)
        subjects.extend([spec.subject] * spec.frame_count)
        actions.extend([spec.action] * spec.frame_count)
        frame_indices.extend(spec.frame_indices)
    return {
        "environment": np.asarray(environments),
        "sample": np.asarray(subjects),
        "action": np.asarray(actions),
        "frame_idx": np.asarray(frame_indices, dtype=np.int64),
    }


def build_dataset(args: argparse.Namespace) -> None:
    src_root = args.src.resolve()
    gt_dir = args.gt_dir.resolve() if args.gt_dir is not None else None
    dst_root = args.dst.resolve()
    dst_root.mkdir(parents=True, exist_ok=True)
    complete_path = dst_root / COMPLETE_FILENAME
    if complete_path.is_file():
        if args.resume:
            print(f"Dataset already complete: {dst_root}")
            return
        raise FileExistsError(f"Dataset already complete: {dst_root}")

    specs = scan_trials(src_root, gt_dir)
    total_frames = sum(spec.frame_count for spec in specs)
    train_subjects = set(args.train_subjects)
    if not any(spec.subject in train_subjects for spec in specs):
        raise ValueError("None of --train-subjects occur in the scanned dataset")
    print(f"Found {len(specs)} aligned trials and {total_frames} frames", flush=True)

    state, raw, gt = _load_or_create_state(
        dst_root,
        specs,
        total_frames,
        args.resume,
    )
    _stage_trials(
        dst_root,
        specs,
        train_subjects,
        args.workers,
        state,
        raw,
        gt,
    )
    lower, upper, mean, std = _normalization_stats(state)
    for variant in (*args.extra_normalizations, "global_minmax"):
        _normalize_variant(
            dst_root,
            raw,
            state,
            variant,
            args.chunk_size,
            lower,
            upper,
            mean,
            std,
        )

    metadata = _metadata_arrays(specs)
    np.savez(dst_root / "meta.npz", **metadata)
    gt_values = np.asarray(gt)
    stats = {
        "schema_version": 2,
        "coordinate_policy": "raw_first_two_channels_no_rescale_no_clip",
        "source_gt_channels": [0, 1],
        "mapped_18_to_raw_17": MAPPED_18_TO_RAW_17,
        "synthetic_joint": {
            "index": SYNTHETIC_JOINT_INDEX,
            "operation": "mean",
            "raw_sources": list(SYNTHETIC_JOINT_SOURCES),
        },
        "joint_names": list(JOINT_NAMES),
        "bone_edges": [list(edge) for edge in CANONICAL_BONE_EDGES],
        "gt_coordinate_min": gt_values.min(axis=(0, 1)).tolist(),
        "gt_coordinate_max": gt_values.max(axis=(0, 1)).tolist(),
        "amplitude_train_min": lower,
        "amplitude_train_max": upper,
        "amplitude_train_mean": mean,
        "amplitude_train_std": std,
        "time_packets": TIME_PACKETS,
        "rx_antennas": RX_ANTENNAS,
        "subcarriers": SUBCARRIERS,
        "total_trials": len(specs),
        "total_frames": total_frames,
        "train_subjects": sorted(train_subjects),
        "normalizations": [*args.extra_normalizations, "global_minmax"],
        "spec_fingerprint": _spec_fingerprint(specs),
    }
    _write_json_atomic(dst_root / "stats.json", stats)
    _write_json_atomic(complete_path, {
        "status": "completed",
        "total_trials": len(specs),
        "total_frames": total_frames,
        "spec_fingerprint": _spec_fingerprint(specs),
    })
    raw_path = dst_root / RAW_CSI_FILENAME
    del raw
    if raw_path.is_file():
        raw_path.unlink()
    state_path = dst_root / STATE_FILENAME
    if state_path.is_file():
        state_path.unlink()
    print(f"Delivery memmap completed: {dst_root}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interrupt-resumable Wi-Pose delivery memmap.",
    )
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument(
        "--train-subjects",
        nargs="+",
        default=[f"S{index:02d}" for index in range(1, 11)],
        help="Subjects used only for CSI normalization statistics.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--extra-normalizations",
        nargs="*",
        choices=("global_zscore", "zscore"),
        default=(),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    build_dataset(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
