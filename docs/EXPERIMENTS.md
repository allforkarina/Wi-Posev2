# Delivery Experiment Protocol

## Claims

The proposed architecture has two testable contributions:

1. Ordered spatial-to-temporal axial attention first resolves antenna and
   subcarrier structure, then models packet evolution.
2. The skeleton-aware joint-query decoder retrieves joint-specific CSI
   evidence and propagates information through the audited human graph.

Every ablation changes one mechanism while preserving the data split, seed,
optimizer, loss, epoch budget, and all unrelated model components.

## Performance experiments P1-P6

| ID | Question | Protocol and evidence |
| --- | --- | --- |
| P1 | In-domain accuracy | Train on env1, evaluate `env1_test` for three seeds. |
| P2 | Zero-shot domain transfer | Evaluate the same source checkpoint on `env2_test`. |
| P3 | Few-shot adaptation | Finetune the full model with nested 540, 810, 4050, and 8100 frame sets; evaluate env2 and env1 forgetting. |
| P4 | Split robustness | Repeat the full source model with random-frame and temporal-block16 manifests. |
| P5 | Diagnostic behavior | Report action, environment, joint, joint-group, tail error, geometry, temporal derivative, and collapse metrics. |
| P6 | Efficiency | Measure parameters, partial MAC/FLOP count, latency, FPS, and peak CUDA memory for representative full checkpoints. |

The checkpoint used for every evaluation is selected only by minimum
validation MPJPE.

## Axial encoder ablation

| ID | Variant | Structural hypothesis |
| --- | --- | --- |
| AX0/C2 | Projection only | Tests whether attention itself is necessary. |
| AX1 | Spatial only | Isolates antenna/subcarrier interactions. |
| AX2 | Temporal only | Isolates packet evolution. |
| AX3 | Temporal then spatial | Tests whether the proposed order matters. |
| AX4 | Parallel sum | Tests order against symmetric fusion. |
| AX5 | Parallel concatenation | Controls for extra fusion capacity. |
| AX6/JD3/C3 | Spatial then temporal | Complete proposed encoder. |

## Joint decoder ablation

| ID | Variant | Structural hypothesis |
| --- | --- | --- |
| JD0/C1 | Global MLP | Removes joint-specific retrieval and graph structure. |
| JD1 | Joint queries plus cross-attention | Tests joint-specific CSI retrieval alone. |
| JD2 | JD1 plus joint self-attention | Adds unconstrained joint interaction. |
| JD3/AX6/C3 | JD2 plus canonical graph | Complete skeleton-aware decoder. |
| JD4 | Shuffled isomorphic graph | Controls for graph computation without correct anatomy. |
| JD5 | Identity graph | Controls for graph parameters without neighbor propagation. |

The combination baseline C0 removes both innovations (`none` axial mode plus
MLP decoder). C1, C2, and C3 reuse JD0, AX0, and the full model respectively,
so no identical architecture is retrained under a second name.

## Optional follow-on diagnostics

These controls are retained for later diagnosis but are not part of the
default 54-run delivery matrix.

| ID | Hypothesis | Matched comparison | Primary evidence |
| --- | --- | --- | --- |
| D4 | Aligning source/target axial-feature covariance can reduce CSI environment shift without erasing pose structure. | `finetune_align --align-loss none --align-weight 0` versus `finetune_align --align-loss coral --align-weight <w>` with identical source checkpoint, manifest, few-shot indices, seed, schedule, losses, and trainable groups. | Target/source MPJPE, PCK, forgetting, per-action errors, feature distributions, and collapse diagnostics. |
| L1 | Uniform coordinate averaging may under-supervise the custom-schema hip, knee, and ankle joints. | `--joint-loss-preset uniform` versus `--joint-loss-preset lower_limb --lower-limb-weight <w>` with every other setting fixed. | Overall and lower-limb MPJPE/PCK, bone errors, and per-joint standard-deviation ratio. |

`finetune_align` uses supervised source replay with weight 1.0, supervised
few-shot target loss, and the optional CORAL term. It uses the same validation
loader and minimum-validation-MPJPE checkpoint rule as ordinary finetuning.
The lower-limb preset uses custom indices `16,5,2,15,14,17`; it does not use
OpenPose or COCO ordering. Both extensions default to a neutral setting, so
existing delivery commands and checkpoint reconstruction remain unchanged.

## Metrics

No pixel-space or original-image-size metric is used. Coordinates remain in
the dataset's compressed scale.

- Accuracy: MPJPE, median/P90/P95 error, coordinate RMSE, X/Y MAE.
- Normalized accuracy: PCK 0.05-0.5 and PCK AUC using the mean of the two
  cross-body shoulder-to-opposite-hip diagonals.
- Alignment diagnostics: N-MPJPE, root-relative MPJPE, and 2D PA-MPJPE.
- Structure: absolute and relative bone-length error, bone-direction error,
  symmetry error, invalid-skeleton rate.
- Localization: per-joint and custom joint-group metrics from
  `data/pose_schema.py`.
- Robustness: per-action macro evidence, zero-shot domain gap, adaptation
  curve, source forgetting, and random-frame versus temporal-block results.
- Dynamics: frame-index-aware velocity and acceleration error.
- Mechanism evidence: spatial attention, temporal attention, action
  difference, and joint-query-to-token attention.
- Statistics: three-seed mean, sample standard deviation, Student-t 95%
  confidence interval, and paired full-minus-control ablation effects.

## Output layout

```text
<output-root>/
  gt_audit/
  seed42|seed123|seed3407/
    manifests/
    random_frame/
      source/<experiment>/
      finetune/<budget>/
    temporal_block16/source/ax6_jd3_c3/
    experiment_registry.csv
    evaluation_index.csv
  summary/
    all_seed_evaluations.csv
    aggregate_metrics.csv
    paired_ablation_effects.csv
    summary.json
```

All evaluations write CSVs. Only seed 42 representative comparisons create
pose composites (random frame, at most two subjects per action, 150 DPI) and
compact attention evidence. After successful postprocessing, each task keeps
only `best_val_mpjpe.pth`.

## Server launch

Run this as one physical shell line:

```bash
cd /data/WiFiPose/Wi-Posev2 && bash scripts/experiments/run_delivery_experiments.sh --project-root /data/WiFiPose/Wi-Posev2 --raw-dataset-root /data/WiFiPose/dataset/dataset --ground-truth-root /data/WiFiPose/dataset/ground_truth_npy --dataset-root /data/WiFiPose/dataset/mmfi_pose_delivery_v1 --output-root /data/WiFiPose/Wi-Posev2/outputs/delivery_v1 --python /data/WiFiPose/Wi-Posev2/.venv/bin/python --workers 4 --batch-size 64
```

The suite is strictly serial on one GPU. `Ctrl+C` is safe: rerunning the same
command resumes memmap staging/normalization and skips completed experiments.
An interrupted in-progress training task restarts from its beginning; already
completed tasks and postprocessing outputs are retained.
