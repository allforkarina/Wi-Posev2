# Final Report Experiments Design

**Date:** 2026-07-02  
**Status:** Approved for implementation planning  
**Branch:** `codex/release2-physical-csi`

## 1. Objective and scope

Build a reproducible experiment system for the final report that evaluates
Wi-Pose under two data-splitting protocols and reruns all Section 2.4
architecture ablations and Section 2.5 cross-domain finetuning experiments from
scratch.

The implementation covers:

- deterministic random-frame and temporal-block 8:1:1 split manifests;
- nested target-domain few-shot manifests covering every subject and action;
- a conventional fully connected coordinate decoder;
- four axial-attention modes, three decoder types, and bone-loss ablation;
- layer-wise finetuning and few-shot data-volume experiments;
- a Wi-Pose-only accuracy and efficiency benchmark;
- a resumable single-seed experiment runner producing exactly 30 training runs.

Pose-relational supervision from Section 2.6 is explicitly outside this phase.
Existing local experiment outputs are not reused.

## 2. Experimental principles

Every comparison must keep the dataset root, split manifest, seed, optimizer,
scheduler, batch size, epoch count, normalization, checkpoint-selection rule,
and evaluation set fixed unless that factor is the experiment variable.

The first pass uses `seed=42` only. The final report must label these results as
single-seed screening results until selected configurations are repeated with
additional seeds.

Configuration selection uses validation metrics only. Test metrics must not be
used to select the axial mode, decoder, trainable group, or few-shot
configuration.

## 3. P0 split manifests

### 3.1 Manifest files

Add a split-manifest module and builder:

```text
data/split_manifest.py
scripts/build_split_manifests.py
```

The builder reads `meta.npz` and produces:

```text
splits/random_frame_seed42.npz
splits/temporal_block16_seed42.npz
```

Both manifests use absolute row indices into the memmap dataset and contain:

```text
env1_train
env1_val
env1_test
env2_train
env2_val
env2_test
env2_fewshot_540
env2_fewshot_810
env2_fewshot_4050
env2_fewshot_8100
```

A sidecar JSON file records the split mode, seed, block size, counts, the
SHA-256 fingerprint of `meta.npz`, SHA-256 hashes of each index array, and
source-train normalization statistics.

### 3.2 Random-frame protocol

Frames are grouped by `(environment, subject, action)`. Each group is shuffled
with a stable seed derived from SHA-256 of the global seed and group identity,
then split approximately 80:10:10. For a group of size `N`, validation and test
counts are each `round(0.1 * N)` and training receives the remainder. For a
297-frame group, the allocation is 237 train, 30 validation, and 30 test frames.

This protocol measures cross-environment performance on unseen frames while
keeping all subjects and actions represented in every split.

### 3.3 Temporal-block protocol

Frames are grouped and sorted as above, divided into consecutive 16-frame
blocks, and the blocks are deterministically shuffled. For `K` blocks,
validation and test receive `max(1, round(0.1 * K))` blocks each and training
receives the remainder. A block cannot cross split boundaries.

This is a distributed temporal-block split, not a single early/middle/late
segment split. It tests whether conclusions survive when temporally adjacent
frames are kept together.

### 3.4 Nested few-shot subsets

Few-shot samples are selected only from `env2_train`. Every target-domain
subject-action group contributes the same number of frames:

| Key | Frames per subject-action | Total frames | Fraction of 80,190 env2 frames |
| --- | ---: | ---: | ---: |
| `env2_fewshot_540` | 2 | 540 | 0.67% |
| `env2_fewshot_810` | 3 | 810 | 1.01% |
| `env2_fewshot_4050` | 15 | 4,050 | 5.05% |
| `env2_fewshot_8100` | 30 | 8,100 | 10.10% |

Each group has one deterministic candidate ordering so the sets satisfy:

```text
540 subset 810 subset 4050 subset 8100
```

Validation and test indices never appear in a few-shot subset.

### 3.5 Normalization

Source normalization statistics are computed from the corresponding
`env1_train` indices only. Because the current global min-max memmap is an
affine transformation of raw amplitude, the builder can store train-only min
and max values in the existing normalized coordinate system. Dataset loading
then applies a second affine transform:

```text
x_split = (x_stored - train_min) / (train_max - train_min)
```

The same source-train transformation is used for env1 validation/test and all
env2 samples. Random-frame and temporal-block protocols have independent
source-train statistics.

### 3.6 Training and evaluation integration

`MemmapDataset` and loader factories accept explicit absolute indices from a
manifest. Existing behavior remains available when no manifest is supplied,
but the final-report runner always requires one.

`train.py` gains narrow manifest arguments sufficient to select source splits,
target validation, and a named few-shot key. `eval.py` gains manifest-backed
`val` and `test` selection. Checkpoints record the split mode, manifest path,
manifest hash, and few-shot key in `train_config`.

## 4. Conventional fully connected decoder

### 4.1 Architecture

Add:

```text
models/wiflow_mlp_decoder.py
```

The decoder maps encoder output to joint coordinates using:

```text
[B, 256, 29, 16]
    -> global average pooling
[B, 256]
    -> fully connected MLP
[B, 36]
    -> reshape
[B, 18, 2]
```

The MLP is fixed as `Linear(256, 1536) -> GELU -> Dropout(0.1) ->
Linear(1536, 1024) -> GELU -> Dropout(0.1) -> Linear(1024, 36)`. It has
approximately 2.006 million parameters versus approximately 1.949 million in
the Joint decoder, a difference of about 2.9%. This reduces the capacity
confound while retaining the defining baseline properties: no joint queries,
no CSI-token cross-attention, no graph propagation, and no joint
self-attention.

The model accepts decoder types:

```text
mlp
joint
hierarchical
```

The selected type is saved in checkpoints and reconstructed by evaluation and
benchmark scripts. MLP checkpoints do not support decoder-latent structure loss;
that loss is outside the current experiment scope.

### 4.2 Scientific interpretation

- MLP tests whether one global CSI vector is sufficient for coordinate
  regression.
- Joint tests whether each joint benefits from selecting different CSI tokens
  before skeleton-aware refinement.
- Hierarchical tests whether staged body-core, limb, and facial retrieval adds
  value beyond parallel joint retrieval.

Performance differences are interpreted together with parameter count, MACs,
latency, per-joint error, and bone error. Accuracy alone cannot establish that
structure rather than capacity caused the difference.

## 5. Section 2.4 architecture experiments

Each split protocol runs seven unique source-only training configurations.

### 5.1 Axial-attention order

| ID | Axial mode | Decoder | Bone weight | Hypothesis |
| --- | --- | --- | ---: | --- |
| A1 | `spatial_then_temporal` | `joint` | 0.5 | Select antenna/subcarrier evidence before temporal motion relations. |
| A2 | `temporal_then_spatial` | `joint` | 0.5 | Motion relations should guide later spatial-frequency selection. |
| A3 | `parallel_sum` | `joint` | 0.5 | Independent spatial and temporal evidence can be fused additively. |
| A4 | `parallel_concat` | `joint` | 0.5 | Preserving both branches before projection avoids destructive summation. |

The comparison reports validation and test PCK, MPJPE, bone error, parameters,
MACs, latency, and attention maps. It determines which ordering is supported by
performance and mechanism evidence; it does not assume the default ordering is
correct in advance.

### 5.2 Decoder structure

| ID | Axial mode | Decoder | Bone weight | Hypothesis |
| --- | --- | --- | ---: | --- |
| D1 | `spatial_then_temporal` | `mlp` | 0.5 | Global pooled CSI is sufficient without joint-specific selection. |
| D2 | `spatial_then_temporal` | `joint` | 0.5 | Joint queries retrieve joint-specific evidence from CSI tokens. |
| D3 | `spatial_then_temporal` | `hierarchical` | 0.5 | Coarse-to-fine context improves structurally dependent joints. |

D2 is identical to A1 and is not retrained.

### 5.3 Bone-loss prior

| ID | Axial mode | Decoder | Bone weight | Hypothesis |
| --- | --- | --- | ---: | --- |
| B1 | `spatial_then_temporal` | `joint` | 0.0 | Coordinate supervision alone is sufficient. |
| B2 | `spatial_then_temporal` | `joint` | 0.5 | Bone supervision reduces structurally implausible predictions. |

B2 is identical to A1 and is not retrained. Evidence includes bone error,
overall and per-joint metrics, and matched skeleton visualizations. A lower
training loss alone is not evidence for the hypothesis.

Every source checkpoint is evaluated on its protocol's env1 validation/test and
env2 validation/test. Env2 is zero-shot at this stage.

## 6. Section 2.5 finetuning experiments

All finetuning starts from the protocol-specific A1 source checkpoint. No old
checkpoint or result directory is reused.

### 6.1 Layer-wise finetuning at 540 target frames

| ID | Trainable group | Approximate trainable fraction | Hypothesis |
| --- | --- | ---: | --- |
| F1 | `spatial_encoder` | 22.1% | Domain shift primarily changes antenna/subcarrier responses. |
| F2 | `axial_encoder` | 6.1% | Domain shift primarily changes learned spatio-temporal relations. |
| F3 | `encoder` | 28.2% | Low-level and axial representations must adapt jointly. |
| F4 | `decoder` | 71.8% | The main mismatch is CSI-feature-to-pose readout. |
| F5 | `full` | 100% | Updating all parameters provides the adaptation upper bound. |

The zero-shot A1 checkpoint is F0 and requires evaluation but no training.
All F1-F5 runs use the exact same `env2_fewshot_540` indices.

Results include PCK, MPJPE, bone error, prediction variance ratio, per-joint and
per-action metrics, and PCK gain per million trainable parameters.

### 6.2 Few-shot data volume

After F1-F5, the runner evaluates every checkpoint on `env2_val` and selects the
trainable group with highest PCK@0.2, using lower MPJPE as the tie-breaker. That
group is then trained with:

| ID | Few-shot key |
| --- | --- |
| V1 | `env2_fewshot_540` |
| V2 | `env2_fewshot_810` |
| V3 | `env2_fewshot_4050` |
| V4 | `env2_fewshot_8100` |

V1 is the corresponding F1-F5 run and is not retrained. The data-volume
experiment therefore adds three training runs per split protocol.

The comparison measures data efficiency, diminishing returns, and whether the
custom difficult joints 10 and 12 require more target supervision than the
full-body average suggests.

## 7. Wi-Pose benchmark

Add:

```text
scripts/benchmark_wipose.py
```

The script loads one Wi-Pose checkpoint, rebuilds its saved architecture,
selects a fixed manifest split, computes accuracy metrics, and measures
efficiency with batch size one.

### 7.1 Accuracy outputs

```text
benchmark_summary.csv
per_joint_metrics.csv
per_action_metrics.csv
per_environment_metrics.csv
per_joint_diagnostic.csv
```

The summary contains PCK@0.1 through PCK@0.5, MPJPE, bone error, sample count,
prediction variance ratio, and mean-pose distance. Existing evaluation metric
definitions are reused rather than reimplemented independently.

### 7.2 Efficiency outputs

```text
runtime_metrics.csv
```

The script reports:

- total and trainable parameters;
- estimated MACs and FLOPs for Conv2d, Linear, and MultiheadAttention;
- batch-1 mean, median, and P95 latency;
- FPS derived from mean latency;
- peak CUDA memory.

MAC estimates include attention Q/K/V projections, attention score/value
products, and output projection. FLOPs are reported as twice MACs. Normalization,
activation, indexing, and elementwise operations are excluded and this
limitation is written into the output metadata.

CUDA measurements use warmup iterations, CUDA events, synchronization, and peak
memory reset. CPU timing uses `perf_counter`. Warmup and measurement iteration
counts are CLI arguments.

Both random-frame and temporal-block checkpoints are benchmarked on their own
fixed test manifests.

## 8. Experiment runner

Add:

```text
scripts/run_report_experiments.py
```

The runner invokes training, evaluation, and benchmark entrypoints with argument
lists through the active Python interpreter. It supports:

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

### 8.1 Execution order per split

```text
1. Seven source-only architecture runs
2. Source validation, env1 test, and env2 zero-shot evaluation
3. Five layer-wise 540-frame finetuning runs
4. Target validation and selection of the best trainable group
5. Three additional data-volume finetuning runs
6. Target test evaluation for every finetuned checkpoint
7. Accuracy and efficiency benchmark for every checkpoint
```

The random-frame protocol completes before the temporal-block protocol.

### 8.2 Run count

Per split protocol:

```text
7 source-only + 5 layer-wise finetunes + 3 additional data-volume finetunes
= 15 training runs
```

Across both protocols:

```text
15 x 2 = 30 training runs
```

Evaluation and benchmark invocations do not count as training runs.

### 8.3 Registry and resume behavior

The runner writes `experiment_registry.csv` with experiment ID, split mode,
full argument list, status, timestamps, duration, checkpoint path, manifest hash,
validation metrics, test metrics, and failure text.

`--dry-run` prints and registers exactly 30 unique training tasks without
executing them. `--resume` skips a task only when its completion marker and
checkpoint both exist and the checkpoint can be loaded with the expected
manifest hash. Partial or corrupt outputs are rerun. The runner is fail-fast by
default; `--continue-on-error` records a failure and proceeds.

## 9. Output layout

```text
outputs/final_report_seed42/
  manifests/
  random_frame/
    source/
    finetune_540/
    finetune_scale/
    registry/
  temporal_block16/
    source/
    finetune_540/
    finetune_scale/
    registry/
  experiment_registry.csv
```

Each experiment directory contains its checkpoints, per-epoch training CSVs,
validation/test evaluation CSVs, benchmark CSVs, and the saved command/config.
Generated datasets and model outputs remain excluded from Git.

## 10. Error handling and invariants

The implementation fails before training when:

- a manifest metadata fingerprint does not match the dataset;
- split keys overlap or do not cover the expected candidates;
- validation or test indices appear in a few-shot set;
- few-shot sets are not nested;
- a subject-action group lacks enough train frames;
- a checkpoint split hash differs from the requested evaluation manifest;
- an MLP decoder is requested with latent-structure supervision;
- CUDA benchmarking is requested without CUDA availability.

No automatic fallback silently changes a split, decoder, checkpoint, or device.

## 11. Verification plan

Implementation follows test-driven development. Tests must cover:

1. random-frame stratification and 8:1:1 counts;
2. whole-block assignment and no temporal block crossing splits;
3. disjoint train/validation/test sets with complete candidate coverage;
4. all-subject/all-action few-shot coverage and strict nesting;
5. deterministic manifests and metadata fingerprint rejection;
6. train-only normalization and consistent target transformation;
7. MLP decoder shape, parameter budget, backward pass, and checkpoint rebuild;
8. all three decoder CLI/configuration contracts;
9. benchmark metric schema and deterministic operation-count estimates;
10. CUDA-independent CPU benchmark smoke behavior;
11. runner dry-run producing exactly 30 unique training tasks;
12. resume rejecting incomplete or incompatible checkpoints;
13. source, finetune, evaluation, and benchmark manifest integration;
14. the complete existing pytest suite.

Before completion, run the full test suite in the `WiFiPose` Conda environment,
inspect the final Git diff, update `AGENTS.md` with the new workflow, and commit
and push only the implementation files to `codex/release2-physical-csi`.
