# Final Report Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Wi-Pose-only experiment pipeline for the approved Section 2.4 architecture ablations and Section 2.5 layer-wise/few-shot finetuning experiments under random-frame and distributed temporal-block splits.

**Architecture:** A versioned split manifest is the single source of truth for training, validation, testing, normalization, and nested few-shot selection. Existing training/evaluation entrypoints gain narrow manifest arguments, while focused scripts build manifests, benchmark one checkpoint, and orchestrate exactly 30 training runs. Existing behavior remains available when no manifest is supplied.

**Tech Stack:** Python 3.10+, NumPy memmaps, PyTorch, argparse, CSV/JSON, pytest, PowerShell, Conda environment `WiFiPose`.

---

## Scope and success criteria

Implementation is complete only when all of the following are true:

- random-frame and temporal-block manifests are deterministic, disjoint, complete, fingerprinted, and contain nested 540/810/4050/8100 target few-shot subsets;
- source-train normalization is applied consistently to every manifest-backed dataset;
- `mlp`, `joint`, and `hierarchical` decoders train, checkpoint, and rebuild correctly;
- source-only training, finetuning, evaluation, and visualization all accept manifest selections without changing the legacy path;
- the benchmark reports accuracy, diagnostics, parameter count, operation estimates, latency, FPS, and peak GPU memory;
- runner dry-run registers exactly 30 unique training jobs and resume validates both checkpoint and manifest hash;
- `AGENTS.md` documents the implemented commands;
- the complete test suite passes in Conda environment `WiFiPose`;
- changes are committed and pushed only to `codex/release2-physical-csi`.

Section 2.6 pose-relational supervision, external-model benchmarks, additional seeds, and automatic report plotting are out of scope.

## File map

**Create**

- `data/split_manifest.py`: manifest schema, stable split construction, hashing, validation, loading, and train-only normalization statistics.
- `scripts/build_split_manifests.py`: CLI that produces both approved manifests and sidecar JSON files.
- `models/wiflow_mlp_decoder.py`: capacity-matched global-pooling MLP coordinate decoder.
- `evaluation/benchmark.py`: reusable parameter, operation-count, latency, and summary helpers.
- `scripts/benchmark_wipose.py`: one-checkpoint Wi-Pose accuracy/efficiency benchmark CLI.
- `experiments/report_suite.py`: pure experiment-matrix and registry/resume logic.
- `scripts/run_report_experiments.py`: subprocess orchestration CLI.
- `tests/test_split_manifest.py`, `tests/test_wiflow_mlp_decoder.py`, `tests/test_manifest_pipeline.py`, `tests/test_benchmark_wipose.py`, `tests/test_report_experiment_runner.py`: focused tests.

**Modify**

- `data/memmap_dataset.py`: accept explicit absolute indices and optional second affine normalization.
- `dataloader.py`: pass manifest indices/statistics into loaders and add a manifest-backed few-shot loader.
- `models/wiflow_model.py`, `models/__init__.py`: register and construct the MLP decoder.
- `train.py`: manifest CLI/config, checkpoint provenance, source/finetune loader selection, and MLP latent-loss validation.
- `eval.py`: manifest split selection, provenance check, and reusable summary CSV output.
- `evaluation/pose_viz.py`: preserve the selected manifest dataset for visualization.
- `AGENTS.md`: document new modules, decoder type, and final-report commands.

No generated manifest, checkpoint, result, or dataset file is committed.

### Task 1: Implement deterministic split manifests

**Files:**

- Create: `data/split_manifest.py`
- Create: `tests/test_split_manifest.py`

- [ ] **Step 1: Write failing tests for random-frame allocation and determinism**

Create synthetic metadata with two environments, two subjects, two actions, and ten ordered frames per group. Test that each group contributes 8/1/1 frames, every index occurs exactly once, and rebuilding produces identical arrays and hashes.

```python
def test_random_frame_split_is_stratified_complete_and_deterministic() -> None:
    meta = synthetic_meta(frames_per_group=10)
    first = build_split_arrays(meta, mode="random_frame", seed=42, block_size=16)
    second = build_split_arrays(meta, mode="random_frame", seed=42, block_size=16)

    for env in ("env1", "env2"):
        assert len(first[f"{env}_train"]) == 32
        assert len(first[f"{env}_val"]) == 4
        assert len(first[f"{env}_test"]) == 4
        combined = np.concatenate([
            first[f"{env}_train"], first[f"{env}_val"], first[f"{env}_test"]
        ])
        assert len(np.unique(combined)) == 40
    assert all(np.array_equal(first[key], second[key]) for key in first)
```

- [ ] **Step 2: Run the test and verify the missing-module failure**

Run:

```powershell
conda activate WiFiPose
pytest tests/test_split_manifest.py -v
```

Expected: collection fails because `data.split_manifest` does not exist.

- [ ] **Step 3: Implement stable grouping and random-frame splits**

Define immutable metadata and manifest types plus stable per-group seeding. Use absolute memmap row indices throughout.

```python
@dataclass(frozen=True)
class DatasetMetadata:
    environment: np.ndarray
    subject: np.ndarray
    action: np.ndarray


@dataclass(frozen=True)
class SplitManifest:
    path: Path
    arrays: Mapping[str, np.ndarray]
    sidecar: Mapping[str, Any]
    manifest_hash: str

    @property
    def mode(self) -> str:
        return str(self.sidecar["mode"])

    @property
    def source_train_normalization(self) -> tuple[float, float]:
        return float(self.sidecar["source_train_min"]), float(self.sidecar["source_train_max"])

    def indices(self, key: str) -> np.ndarray:
        if key not in self.arrays:
            raise KeyError(f"Manifest has no split key: {key}")
        return np.asarray(self.arrays[key], dtype=np.int64)


def stable_group_seed(seed: int, group: tuple[str, str, str]) -> int:
    payload = "\0".join((str(seed), *group)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _random_group_split(indices: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    shuffled = np.random.default_rng(seed).permutation(indices)
    val_count = round(0.1 * len(indices))
    test_count = round(0.1 * len(indices))
    train_count = len(indices) - val_count - test_count
    return shuffled[:train_count], shuffled[train_count:train_count + val_count], shuffled[-test_count:]
```

`build_split_arrays()` must group by `(environment, subject, action)`, apply the helper independently, concatenate groups by sorted identity, and sort each final array for reproducible I/O.

- [ ] **Step 4: Add failing temporal-block tests**

```python
def test_temporal_blocks_never_cross_split_boundaries() -> None:
    meta = synthetic_meta(frames_per_group=160)
    arrays = build_split_arrays(meta, mode="temporal_block", seed=42, block_size=16)
    ownership = {
        int(index): split
        for split in ("train", "val", "test")
        for index in arrays[f"env1_{split}"]
    }
    group = group_indices(meta, ("env1", "S01", "A01"))
    for start in range(0, len(group), 16):
        assert len({ownership[int(i)] for i in group[start:start + 16]}) == 1
```

- [ ] **Step 5: Implement distributed temporal-block assignment**

Sort each group by absolute row index, split with `np.array_split`-equivalent consecutive slices of at most 16 frames, deterministically shuffle the block list, assign `max(1, round(0.1*K))` blocks to validation and test, and reject groups with fewer than three blocks because they cannot populate all splits without leakage.

- [ ] **Step 6: Add failing nested few-shot tests**

Use a fixture with at least 30 training frames per `(env2, subject, action)` and assert exact per-group quotas, all-subject/all-action coverage, disjointness from validation/test, and strict set inclusion.

```python
def test_few_shot_sets_are_balanced_nested_and_train_only() -> None:
    arrays = build_split_arrays(synthetic_meta(frames_per_group=64), "random_frame", 42, 16)
    attach_few_shot_arrays(arrays, synthetic_meta(frames_per_group=64), seed=42)
    sizes = [len(arrays[f"env2_fewshot_{n}"]) for n in (16, 24, 120, 240)]
    assert sizes == [16, 24, 120, 240]
    assert set(arrays["env2_fewshot_16"]) < set(arrays["env2_fewshot_24"])
    assert set(arrays["env2_fewshot_24"]) < set(arrays["env2_fewshot_120"])
    assert not set(arrays["env2_fewshot_240"]) & set(arrays["env2_val"])
```

The production quotas remain `(2, 3, 15, 30)` frames per group and keys remain `540`, `810`, `4050`, `8100`; allow quota/key pairs to be injected only in the pure builder so tiny tests do not require the full dataset.

- [ ] **Step 7: Implement manifest validation and hashing**

Add:

```python
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.int64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def validate_manifest(
    arrays: Mapping[str, np.ndarray],
    metadata: DatasetMetadata,
    few_shot_keys: tuple[str, ...] = (
        "env2_fewshot_540",
        "env2_fewshot_810",
        "env2_fewshot_4050",
        "env2_fewshot_8100",
    ),
) -> None:
    required = {f"{env}_{split}" for env in ("env1", "env2") for split in ("train", "val", "test")}
    required.update(few_shot_keys)
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"Manifest is missing keys: {missing}")
    total = len(metadata.environment)
    for key in required:
        values = arrays[key]
        if values.dtype != np.int64:
            raise ValueError(f"Manifest key {key} must use int64 indices")
        if len(values) != len(np.unique(values)):
            raise ValueError(f"Manifest key {key} contains duplicate indices")
        if np.any(values < 0) or np.any(values >= total):
            raise ValueError(f"Manifest key {key} contains out-of-range indices")
    for env in ("env1", "env2"):
        split_sets = {
            split: set(arrays[f"{env}_{split}"].tolist())
            for split in ("train", "val", "test")
        }
        if any(split_sets[left] & split_sets[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError(f"{env} train/val/test indices overlap")
        expected = {i for i, value in enumerate(metadata.environment) if str(value) == env}
        if set.union(*split_sets.values()) != expected:
            raise ValueError(f"{env} train/val/test indices do not cover the environment")
    env2_train = set(arrays["env2_train"].tolist())
    previous: set[int] = set()
    for key in few_shot_keys:
        current = set(arrays[key].tolist())
        if not current <= env2_train:
            raise ValueError(f"Manifest key {key} contains values outside env2_train")
        if not previous <= current:
            raise ValueError(f"Manifest few-shot sets are not nested at {key}")
        previous = current


def save_manifest(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    sidecar: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(
        json.dumps(dict(sidecar), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_manifest(path: Path, dataset_root: Path) -> SplitManifest:
    arrays = dict(np.load(path, allow_pickle=False))
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if sidecar["meta_sha256"] != sha256_file(dataset_root / "meta.npz"):
        raise ValueError("Manifest metadata fingerprint does not match dataset meta.npz")
    for key, expected_hash in sidecar["array_sha256"].items():
        if key not in arrays or sha256_array(arrays[key]) != expected_hash:
            raise ValueError(f"Manifest array hash mismatch for key: {key}")
    return SplitManifest(
        path=path.resolve(),
        arrays=arrays,
        sidecar=sidecar,
        manifest_hash=sha256_file(path),
    )
```

Validation must reject missing keys, non-`int64` or out-of-range values, duplicate indices, train/val/test overlaps, incomplete environment coverage, nonnested few-shot sets, few-shot values outside `env2_train`, mismatched `meta.npz` SHA-256, and mismatched array hashes. Error messages must name the failing key or invariant.

- [ ] **Step 8: Run manifest tests**

Run `pytest tests/test_split_manifest.py -v`.

Expected: all manifest tests pass.

- [ ] **Step 9: Commit the manifest core**

```powershell
git add data/split_manifest.py tests/test_split_manifest.py
git commit -m "Add deterministic experiment split manifests"
```

### Task 2: Compute and apply source-train normalization

**Files:**

- Modify: `data/split_manifest.py`
- Modify: `data/memmap_dataset.py`
- Modify: `dataloader.py`
- Modify: `tests/test_split_manifest.py`
- Create: `tests/test_manifest_pipeline.py`

- [ ] **Step 1: Write failing normalization tests**

Build a tiny `csi_gminmax.npy` whose source-train subset spans `[0.25, 0.75]`. Assert source train maps to `[0, 1]`, while a target value of `1.0` maps to `1.5`; do not clamp target values.

```python
dataset = MemmapDataset(
    data_dir=root,
    split="all",
    indices=np.array([target_index], dtype=np.int64),
    split_normalization=(0.25, 0.75),
)
assert torch.allclose(dataset[0]["csi"], torch.full_like(dataset[0]["csi"], 1.5))
```

- [ ] **Step 2: Verify the new constructor arguments fail**

Run `pytest tests/test_manifest_pipeline.py -v`.

Expected: `MemmapDataset.__init__()` rejects `indices`.

- [ ] **Step 3: Add manifest normalization statistics**

Compute scalar minimum and maximum by iterating `csi_gminmax.npy[env1_train]` in bounded chunks, record them as `source_train_min` and `source_train_max` in the JSON sidecar, and reject a range smaller than `1e-12`.

- [ ] **Step 4: Add explicit-index dataset loading**

Extend `MemmapDataset.__init__` with:

```python
indices: Iterable[int] | np.ndarray | None = None,
split_normalization: tuple[float, float] | None = None,
```

When `indices` is provided, validate and copy it as `int64` and bypass `_build_split`. In `__getitem__`, after copying CSI, apply:

```python
if self.split_normalization is not None:
    lower, upper = self.split_normalization
    csi = (csi - lower) / (upper - lower)
```

Do not clamp and do not mutate the memmap.

- [ ] **Step 5: Add narrow manifest loader factories**

Extend `create_memmap_data_loader` with optional `indices` and `split_normalization`. Add:

```python
def create_manifest_data_loader(
    data_dir: str | Path,
    manifest: SplitManifest,
    key: str,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    return create_memmap_data_loader(
        data_dir=data_dir,
        split="all",
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        indices=manifest.indices(key),
        split_normalization=manifest.source_train_normalization,
    )
```

Keep all legacy call signatures operational.

- [ ] **Step 6: Verify dataset/loader behavior**

Run:

```powershell
pytest tests/test_split_manifest.py tests/test_manifest_pipeline.py tests/test_memmap_dataset.py tests/test_dataloader.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit normalization integration**

```powershell
git add data/split_manifest.py data/memmap_dataset.py dataloader.py tests/test_split_manifest.py tests/test_manifest_pipeline.py
git commit -m "Apply manifest train-only normalization"
```

### Task 3: Add the manifest builder CLI

**Files:**

- Create: `scripts/build_split_manifests.py`
- Modify: `tests/test_split_manifest.py`

- [ ] **Step 1: Write a failing CLI smoke test**

Call `main(["--dataset-root", str(root), "--output-dir", str(output), "--seed", "42", "--block-size", "16"])` with a tiny valid memmap fixture and a monkeypatched small test quota table. Assert two `.npz` and two `.json` files exist and reload successfully.

- [ ] **Step 2: Implement the CLI**

Expose:

```text
--dataset-root PATH
--output-dir PATH
--seed 42
--block-size 16
```

`main(argv: Sequence[str] | None = None)` loads `meta.npz`, builds `random_frame_seed42.npz` and `temporal_block16_seed42.npz`, computes normalization, validates each saved file by reloading it, and prints counts. Production CLI always uses approved keys/quotas.

- [ ] **Step 3: Run CLI tests and help output**

```powershell
pytest tests/test_split_manifest.py -v
python scripts/build_split_manifests.py --help
```

Expected: tests pass and help lists all four arguments.

- [ ] **Step 4: Commit the builder**

```powershell
git add scripts/build_split_manifests.py tests/test_split_manifest.py
git commit -m "Add split manifest builder CLI"
```

### Task 4: Add the conventional MLP decoder

**Files:**

- Create: `models/wiflow_mlp_decoder.py`
- Modify: `models/wiflow_model.py`
- Modify: `models/__init__.py`
- Create: `tests/test_wiflow_mlp_decoder.py`

- [ ] **Step 1: Write failing shape, parameter, gradient, and model tests**

```python
def test_mlp_decoder_shape_parameter_budget_and_backward() -> None:
    decoder = WiFlowMLPDecoder()
    features = torch.randn(2, 256, 29, 16, requires_grad=True)
    coordinates = decoder(features)
    assert coordinates.shape == (2, 18, 2)
    assert 1_950_000 <= sum(p.numel() for p in decoder.parameters()) <= 2_050_000
    coordinates.sum().backward()
    assert features.grad is not None


def test_mlp_decoder_has_no_joint_latent_contract() -> None:
    with pytest.raises(ValueError, match="does not expose joint latent features"):
        WiFlowMLPDecoder()(torch.randn(1, 256, 29, 16), return_features=True)
```

Also assert `WiFlowModel(decoder_type="mlp")` forwards `[B,3,114,64]` to `[B,18,2]` and `DECODER_TYPES == ("mlp", "joint", "hierarchical")`.

- [ ] **Step 2: Run tests and verify import failure**

Run `pytest tests/test_wiflow_mlp_decoder.py -v`.

- [ ] **Step 3: Implement the exact approved decoder**

```python
class WiFlowMLPDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_dim = 256
        self.coordinate_head = nn.Sequential(
            nn.Linear(256, 1536),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1536, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, NUM_OPENPOSE_KEYPOINTS * 2),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.embedding_dim:
            raise ValueError("WiFlowMLPDecoder expects input shaped [B, 256, H, W]")
        if return_features:
            raise ValueError("MLP decoder does not expose joint latent features")
        pooled = x.mean(dim=(-2, -1))
        return self.coordinate_head(pooled).reshape(x.shape[0], NUM_OPENPOSE_KEYPOINTS, 2)
```

Construct it in `WiFlowModel` and export it from `models/__init__.py`. Do not change the joint or hierarchical decoder.

- [ ] **Step 4: Verify decoder tests and existing model contracts**

```powershell
pytest tests/test_wiflow_mlp_decoder.py tests/test_wiflow_model.py tests/test_wiflow_decoder.py -v
```

- [ ] **Step 5: Commit the decoder**

```powershell
git add models/wiflow_mlp_decoder.py models/wiflow_model.py models/__init__.py tests/test_wiflow_mlp_decoder.py
git commit -m "Add conventional MLP pose decoder"
```

### Task 5: Integrate manifests into training and evaluation

**Files:**

- Modify: `train.py`
- Modify: `eval.py`
- Modify: `evaluation/pose_viz.py`
- Modify: `tests/test_manifest_pipeline.py`
- Modify: `tests/test_wiflow_mlp_decoder.py`

- [ ] **Step 1: Write failing CLI/configuration tests**

Assert training parses:

```text
--split-manifest PATH
--few-shot-key env2_fewshot_540
```

Assert evaluation parses:

```text
--split-manifest PATH
--manifest-key env1_test
```

Assert `TrainConfig` stores `split_manifest` and `few_shot_key`, and that MLP plus nonzero latent-structure weight raises before model construction.

- [ ] **Step 2: Add training configuration and early validation**

Add nullable fields:

```python
split_manifest: str | None = None
few_shot_key: str | None = None
```

Add:

```python
def validate_training_config(config: TrainConfig) -> None:
    if config.decoder_type == "mlp" and config.latent_structure_loss_weight > 0:
        raise ValueError("MLP decoder cannot use latent-structure supervision")
    if config.mode == "finetune" and config.split_manifest and not config.few_shot_key:
        raise ValueError("Manifest-backed finetuning requires --few-shot-key")
```

- [ ] **Step 3: Route source-only loaders through manifest keys**

When a manifest is supplied, load and fingerprint-check it once and create:

```python
train_loader = create_manifest_data_loader(
    data_dir=config.dataset_root,
    manifest=manifest,
    key="env1_train",
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    shuffle=True,
)
val_loader = create_manifest_data_loader(
    data_dir=config.dataset_root,
    manifest=manifest,
    key="env1_val",
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    shuffle=False,
)
test_loader = create_manifest_data_loader(
    data_dir=config.dataset_root,
    manifest=manifest,
    key="env1_test",
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    shuffle=False,
)
```

Preserve the current loader path when `split_manifest is None`.

- [ ] **Step 4: Route finetuning through one named few-shot set**

For manifest-backed finetuning, use `config.few_shot_key` as training indices and `env2_val` for checkpoint selection. Save the absolute few-shot values to `few_shot_train_indices.npy`. Source replay, when enabled, uses `env1_train` from the same manifest.

- [ ] **Step 5: Save checkpoint provenance**

Add these values to `train_config` for every checkpoint:

```python
"split_manifest": str(manifest.path.resolve()) if manifest else None,
"split_mode": manifest.mode if manifest else None,
"manifest_hash": manifest.manifest_hash if manifest else None,
"few_shot_key": config.few_shot_key,
```

The hash is the SHA-256 of the `.npz` bytes; it is not a machine-dependent path hash.

- [ ] **Step 6: Add evaluation manifest selection and provenance validation**

Add arguments `--split-manifest` and `--manifest-key`. If either is provided, require both. Load the exact absolute indices and normalization from the manifest. Reject a checkpoint whose non-null `manifest_hash` differs from the loaded manifest hash.

Write `benchmark_summary.csv` with one row containing `sample_count`, MPJPE, bone error, PCK@0.1–0.5, overall variance ratio, and mean-pose distance. Keep existing per-joint/action/environment/diagnostic CSVs.

- [ ] **Step 7: Keep pose visualization on the same selected dataset**

Pass the already selected manifest-backed dataset to `run_pose_visualization`; do not reconstruct an unfiltered `split="all"` dataset. Feature visualization continues to use the selected loader.

- [ ] **Step 8: Add checkpoint rebuild coverage for MLP**

Save a tiny MLP model checkpoint with `decoder_type="mlp"`, load it with `eval.load_checkpoint_model`, and assert the rebuilt type and output shape.

- [ ] **Step 9: Run integration tests**

```powershell
pytest tests/test_manifest_pipeline.py tests/test_wiflow_mlp_decoder.py tests/test_training_metric_artifacts.py tests/test_eval_png_only.py -v
```

Expected: all pass, including legacy nonmanifest behavior.

- [ ] **Step 10: Commit training/evaluation integration**

```powershell
git add train.py eval.py evaluation/pose_viz.py tests/test_manifest_pipeline.py tests/test_wiflow_mlp_decoder.py
git commit -m "Integrate manifests with training and evaluation"
```

### Task 6: Implement Wi-Pose accuracy and efficiency benchmark

**Files:**

- Create: `evaluation/benchmark.py`
- Create: `scripts/benchmark_wipose.py`
- Create: `tests/test_benchmark_wipose.py`

- [ ] **Step 1: Write failing parameter and MAC tests**

Test exact formulas on tiny modules:

```python
def test_linear_mac_count() -> None:
    layer = nn.Linear(4, 3, bias=True)
    assert estimate_module_macs(layer, (2, 5, 4), (2, 5, 3)) == 2 * 5 * 4 * 3


def test_conv2d_mac_count() -> None:
    layer = nn.Conv2d(3, 8, kernel_size=3, groups=1)
    assert estimate_module_macs(layer, (1, 3, 10, 10), (1, 8, 8, 8)) == 1 * 8 * 8 * 8 * 3 * 3 * 3
```

For `nn.MultiheadAttention`, count Q/K/V projections, score products, value products, and output projection using batch, query length, key length, embed dimension, and head dimension. Ensure child Linear hooks are not double-counted for MHA internals.

- [ ] **Step 2: Implement operation counting**

Use temporary forward hooks only on `Conv2d`, `Linear`, and `MultiheadAttention`. Return an immutable result containing `macs`, `flops=2*macs`, and the limitation string:

```text
Counts Conv2d, Linear, and MultiheadAttention only; excludes normalization,
activation, indexing, pooling, and elementwise operations.
```

- [ ] **Step 3: Write failing CPU latency tests**

Patch a tiny model, run two warmups and five measurements, and assert positive mean/median/P95 latency, `fps == 1000 / mean_latency_ms`, and `peak_cuda_memory_mb == 0` on CPU.

- [ ] **Step 4: Implement latency measurement**

CPU uses `time.perf_counter()` around each forward. CUDA requires availability, resets peak memory, synchronizes before/after warmup, measures each iteration with CUDA events, and records `torch.cuda.max_memory_allocated()`. Never silently fall back from CUDA to CPU.

- [ ] **Step 5: Implement the benchmark CLI**

Arguments:

```text
--dataset-root PATH
--checkpoint PATH
--split-manifest PATH
--manifest-key KEY
--output-dir PATH
--batch-size 64
--device cuda
--warmup-iterations 20
--measure-iterations 100
--num-workers 0
```

The CLI must:

1. load and provenance-check the checkpoint/manifest;
2. reuse `run_evaluation()` and the existing CSV writers for accuracy;
3. run batch-1 efficiency measurement on input `[1,3,114,64]`;
4. write `benchmark_summary.csv`, all four detailed evaluation CSVs, and `runtime_metrics.csv`;
5. include total/trainable parameters, MACs/FLOPs, mean/median/P95 milliseconds, FPS, peak CUDA MB, warmup/measurement counts, device, and limitation text.

- [ ] **Step 6: Run benchmark tests and CPU smoke CLI**

```powershell
pytest tests/test_benchmark_wipose.py -v
python scripts/benchmark_wipose.py --help
```

Expected: all tests pass and CLI help is complete.

- [ ] **Step 7: Commit benchmark support**

```powershell
git add evaluation/benchmark.py scripts/benchmark_wipose.py tests/test_benchmark_wipose.py
git commit -m "Add Wi-Pose accuracy and efficiency benchmark"
```

### Task 7: Define the exact 30-run experiment matrix

**Files:**

- Create: `experiments/__init__.py`
- Create: `experiments/report_suite.py`
- Create: `tests/test_report_experiment_runner.py`

- [ ] **Step 1: Write a failing matrix test**

```python
def test_report_suite_contains_exactly_thirty_unique_training_tasks() -> None:
    tasks = build_training_tasks(config_for_test())
    assert len(tasks) == 30
    assert len({task.experiment_id for task in tasks}) == 30
    assert [task.split_mode for task in tasks[:15]] == ["random_frame"] * 15
    assert [task.split_mode for task in tasks[15:]] == ["temporal_block"] * 15
    assert sum(task.phase == "source" for task in tasks) == 14
    assert sum(task.phase == "finetune_540" for task in tasks) == 10
    assert sum(task.phase == "finetune_scale" for task in tasks) == 6
```

- [ ] **Step 2: Implement immutable task/config records**

```python
@dataclass(frozen=True)
class SuiteConfig:
    dataset_root: Path
    output_root: Path
    split_modes: tuple[str, ...]
    seed: int
    source_epochs: int
    finetune_epochs: int
    batch_size: int
    device: str


@dataclass(frozen=True)
class ExperimentTask:
    experiment_id: str
    split_mode: str
    phase: str
    command: tuple[str, ...]
    output_dir: Path
    manifest_path: Path
```

- [ ] **Step 3: Encode seven source runs per split**

Generate A1–A4, D1, D3, and B1 only. D2/B2 are aliases of A1 in report metadata, never extra training tasks. Every command fixes seed, epochs, batch size, device, source env, manifest, and output directory.

- [ ] **Step 4: Encode five 540-frame finetunes per split**

Generate F1–F5 for `spatial_encoder`, `axial_encoder`, `encoder`, `decoder`, and `full`, all from the split-specific A1 `best_val_pck_0_2.pth` and all using `env2_fewshot_540`.

- [ ] **Step 5: Represent three selected-group scale tasks per split**

V2–V4 use keys `env2_fewshot_810`, `env2_fewshot_4050`, `env2_fewshot_8100`. Their trainable group is resolved only after F1–F5 validation. Keep them in dry-run as symbolic commands with token `{selected_trainable_group}` so the planned count remains 30; replace the token before real execution and store the resolved command in the registry.

- [ ] **Step 6: Test command invariants**

Assert no source task enables a Section 2.6 loss, all tasks use seed 42, all finetune tasks point to A1, all split paths are protocol-specific, and no output directory repeats.

- [ ] **Step 7: Run matrix tests**

Run `pytest tests/test_report_experiment_runner.py -v`.

- [ ] **Step 8: Commit the matrix**

```powershell
git add experiments/__init__.py experiments/report_suite.py tests/test_report_experiment_runner.py
git commit -m "Define final report experiment matrix"
```

### Task 8: Implement registry, selection, resume, and orchestration

**Files:**

- Modify: `experiments/report_suite.py`
- Create: `scripts/run_report_experiments.py`
- Modify: `tests/test_report_experiment_runner.py`

- [ ] **Step 1: Write failing validation-selection tests**

Create five synthetic F1–F5 `benchmark_summary.csv` files. Assert highest `pck_0_2` wins and lower MPJPE breaks exact PCK ties. Missing/invalid validation files must fail rather than select from test metrics.

- [ ] **Step 2: Implement validation-only group selection**

```python
def select_trainable_group(rows: Sequence[Mapping[str, str]]) -> str:
    ranked = sorted(
        rows,
        key=lambda row: (-float(row["pck_0_2"]), float(row["mpjpe"]), row["experiment_id"]),
    )
    return ranked[0]["trainable_group"]
```

Require exactly one valid row for every F1–F5 experiment.

- [ ] **Step 3: Write failing resume tests**

Assert skip occurs only when all exist and agree:

- `completed.json` with status `completed`;
- expected checkpoint path;
- loadable checkpoint dictionary with `model_state_dict` and `train_config`;
- checkpoint `manifest_hash` equals current manifest hash.

Missing marker, corrupt checkpoint, failed status, or hash mismatch returns rerun.

- [ ] **Step 4: Implement completion and registry helpers**

Use atomic replacement for `completed.json` and `experiment_registry.csv`. Registry columns are:

```text
experiment_id,split_mode,phase,command,status,started_at,finished_at,
duration_seconds,checkpoint_path,manifest_hash,val_pck_0_2,val_mpjpe,
test_pck_0_2,test_mpjpe,failure
```

Never infer completion from a directory alone.

- [ ] **Step 5: Implement subprocess execution**

Use `sys.executable` and argument lists, not shell strings. Save `command.json` before launch. Fail fast unless `--continue-on-error`; on failure record return code and stderr tail. Do not catch `KeyboardInterrupt` as an ordinary experiment failure.

- [ ] **Step 6: Implement per-split workflow**

For each split in user-specified order:

1. ensure the required manifest exists or invoke the builder;
2. train seven source configurations;
3. evaluate each on `env1_val`, `env1_test`, `env2_val`, and `env2_test`;
4. benchmark each source checkpoint on its manifest-backed test set;
5. train F1–F5 on the same 540 indices;
6. evaluate F1–F5 on `env2_val` and select a group;
7. train V2–V4 using the selected group;
8. evaluate/benchmark all finetuned checkpoints on `env2_test`.

Target test is never read during group selection.

- [ ] **Step 7: Implement CLI and dry-run**

Expose exactly the approved arguments:

```text
--dataset-root
--output-root
--split-modes random_frame temporal_block
--seed 42
--source-epochs 50
--finetune-epochs 30
--batch-size 64
--device cuda
--dry-run
--resume
--continue-on-error
```

`--dry-run` writes planned registry rows and prints exactly 30 training commands without launching training/evaluation/benchmark subprocesses.

- [ ] **Step 8: Verify orchestration tests**

```powershell
pytest tests/test_report_experiment_runner.py -v
python scripts/run_report_experiments.py --help
```

- [ ] **Step 9: Commit the runner**

```powershell
git add experiments/report_suite.py scripts/run_report_experiments.py tests/test_report_experiment_runner.py
git commit -m "Add resumable final report experiment runner"
```

### Task 9: Document the implemented workflow

**Files:**

- Modify: `AGENTS.md`

- [ ] **Step 1: Update project structure and model contracts**

Document `data/split_manifest.py`, both new scripts, `evaluation/benchmark.py`, `experiments/report_suite.py`, manifest-backed normalization, and `mlp|joint|hierarchical` decoder support.

- [ ] **Step 2: Add exact final-report commands**

```powershell
conda activate WiFiPose
python scripts\build_split_manifests.py --dataset-root data\mmfi_pose --output-dir outputs\final_report_seed42\manifests --seed 42 --block-size 16
python scripts\run_report_experiments.py --dataset-root data\mmfi_pose --output-root outputs\final_report_seed42 --split-modes random_frame temporal_block --seed 42 --source-epochs 50 --finetune-epochs 30 --batch-size 64 --device cuda --dry-run
python scripts\run_report_experiments.py --dataset-root data\mmfi_pose --output-root outputs\final_report_seed42 --split-modes random_frame temporal_block --seed 42 --source-epochs 50 --finetune-epochs 30 --batch-size 64 --device cuda --resume
```

Also document one direct manifest-backed `train.py`, `eval.py`, and `benchmark_wipose.py` command for debugging a single run.

- [ ] **Step 3: Verify documentation against CLI help**

Run all four relevant `--help` commands and compare argument spelling with `AGENTS.md`.

- [ ] **Step 4: Commit documentation**

```powershell
git add AGENTS.md
git commit -m "Document final report experiment workflow"
```

### Task 10: Full verification and branch delivery

**Files:**

- Review all files changed in Tasks 1–9.

- [ ] **Step 1: Run focused tests**

```powershell
conda activate WiFiPose
pytest tests/test_split_manifest.py tests/test_manifest_pipeline.py tests/test_wiflow_mlp_decoder.py tests/test_benchmark_wipose.py tests/test_report_experiment_runner.py -v
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the complete suite**

```powershell
pytest
```

Expected: all tests pass with no regressions.

- [ ] **Step 3: Run a 30-task dry-run audit**

Use a tiny valid synthetic dataset fixture or the real dataset if locally available:

```powershell
python scripts\run_report_experiments.py --dataset-root data\mmfi_pose --output-root outputs\final_report_seed42 --split-modes random_frame temporal_block --seed 42 --source-epochs 50 --finetune-epochs 30 --batch-size 64 --device cuda --dry-run
```

Verify 30 unique training rows, 15 per split, and no launched training process. Do not commit the generated registry.

- [ ] **Step 4: Inspect repository state and diff**

```powershell
git status --short
git diff --check
git diff --stat HEAD~9..HEAD
git branch --show-current
```

Expected: no whitespace errors; only scoped source/tests/docs are tracked; branch is `codex/release2-physical-csi`. Preserve unrelated untracked user directories.

- [ ] **Step 5: Create a final verification commit only if verification required fixes**

Inspect `git status --short`, stage only the explicitly listed source/test/documentation files changed to fix a verification failure, then commit them with message `Fix final report experiment verification issues`. Skip this commit when no files changed.

- [ ] **Step 6: Push the current branch**

```powershell
git push origin codex/release2-physical-csi
```

Expected: remote branch advances to the final verified commit.

## Self-review record

- Spec coverage: Tasks 1–3 cover P0 manifests and normalization; Task 4 covers the MLP decoder; Task 5 covers training/evaluation/checkpoint integration; Task 6 covers Wi-Pose-only metrics and efficiency; Tasks 7–8 cover exactly 30 experiments, selection, registry, and resume; Task 9 covers the required workflow documentation; Task 10 covers complete verification and push.
- Scope exclusions remain explicit: no Section 2.6 runs, external models, extra seeds, or result-plot generation.
- Type consistency: manifest keys, `split_manifest`, `few_shot_key`, `manifest_hash`, decoder names, experiment phases, and benchmark column names are used consistently across tasks.
- Placeholder scan: implementation behavior contains no deferred design decisions; verification-only fixes are staged from the explicit paths reported by `git status --short` rather than a wildcard command.
