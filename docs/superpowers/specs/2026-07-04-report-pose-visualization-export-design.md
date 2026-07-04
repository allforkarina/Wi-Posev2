# Final-Report Pose Visualization Export Design

## Objective

Add a standalone Linux-friendly export command that renders reproducible prediction-versus-ground-truth pose PNGs from the completed random-frame report experiments. The exporter must not retrain models or overwrite the existing training, evaluation, or benchmark artifacts.

The export covers five checkpoints:

- Source A1 evaluated on `env1_test`.
- Full finetuning with 540 target frames evaluated on `env2_test`.
- Full finetuning with 810 target frames evaluated on `env2_test`.
- Full finetuning with 4050 target frames evaluated on `env2_test`.
- Full finetuning with 8100 target frames evaluated on `env2_test`.

## Scope

The implementation adds `scripts/export_report_pose_visualizations.py` and a narrow reusable single-sample rendering entry point in `evaluation/pose_viz.py`. It also adds focused tests and documents the command in `AGENTS.md`.

The implementation does not change model architectures, checkpoint contents, training behavior, `eval.py` defaults, experiment metrics, or existing report outputs. It generates PNG files only and never generates PDF files.

## Command-Line Interface

The standalone script accepts:

```text
--dataset-root PATH
--experiment-root PATH
--output-dir PATH
--seed INT                    default: 42
--batch-size INT              default: 64
--num-workers INT             default: 0
--device DEVICE               default: cuda
```

`--experiment-root` is the completed suite root, for example `outputs/final_report_seed42_v4`. The script resolves the random-frame manifest and the following checkpoint paths underneath that root:

```text
manifests/random_frame_seed42.npz
random_frame/source/a1/best_val_pck_0_2.pth
random_frame/finetune_540/f5/best_val_pck_0_2.pth
random_frame/finetune_scale/v2/best_val_pck_0_2.pth
random_frame/finetune_scale/v3/best_val_pck_0_2.pth
random_frame/finetune_scale/v4/best_val_pck_0_2.pth
```

The exporter fails before inference if the manifest, any checkpoint, or required dataset metadata is missing. It also fails if `--output-dir` already exists and is non-empty, preventing accidental replacement of report artifacts.

## Deterministic Sample Selection

The exporter loads the random-frame manifest with the existing dataset fingerprint validation. It constructs the source candidate set from `env1_test` and the target candidate set from `env2_test`.

For each candidate set it:

1. Groups absolute dataset indices by action identifier.
2. Sorts action identifiers to make traversal deterministic.
3. Uses `numpy.random.default_rng(seed)` to select exactly one index from each action group.
4. Validates that every discovered action produced one selected index.

Source and target selections are independent because they belong to different environments. The target selection is computed once and reused unchanged for all four finetuned checkpoints. The exporter does not select frames using model errors, PCK, MPJPE, or visual quality, avoiding result-dependent cherry-picking.

The selected samples are recorded before inference:

```text
sample_indices/env1_test_seed42.csv
sample_indices/env2_test_seed42.csv
```

Each CSV row contains:

```text
action,dataset_index,subject,environment
```

`dataset_index` is the absolute memmap dataset index and is the reproducibility key.

## Inference and Rendering

The script rebuilds each model with the existing checkpoint-loading function, including the saved training configuration and manifest hash check. It performs batched inference only for the selected samples.

`evaluation/pose_viz.py` exposes a public single-sample renderer that reuses the existing joint colors, project-specific skeleton edges, axis orientation, and PNG-only save behavior. The renderer accepts target keypoints, predicted keypoints, sample metadata, an absolute dataset index, and a model label.

Each PNG uses the approved two-panel layout:

- Left: hollow GT joints, filled prediction joints, joint numbers, and dashed error vectors.
- Right: dashed GT skeleton overlaid with a solid prediction skeleton.
- Title: model label, action, subject, environment, and dataset index.
- Resolution: 300 DPI.

Per-action composite figures are not generated because there is exactly one selected sample per action and a composite would duplicate the individual figure.

## Output Layout

```text
<output-dir>/
├── sample_indices/
│   ├── env1_test_seed42.csv
│   └── env2_test_seed42.csv
├── source_a1_env1/
├── finetune_540_env2/
├── finetune_810_env2/
├── finetune_4050_env2/
└── finetune_8100_env2/
```

Within each model directory, images are grouped by action:

```text
<model-label>/<action>/<subject>_<environment>_idx<dataset-index>.png
```

The same target action therefore has the same subject, environment, and dataset index in all four finetuning directories.

## Error Handling

The command raises a clear error and exits non-zero when:

- The dataset root is invalid.
- The manifest is missing or its dataset fingerprint does not match.
- A required checkpoint is missing.
- A required manifest split is absent.
- An action group has no selectable sample.
- Model output and GT shapes do not match OpenPose18 `[18, 2]` coordinates.
- The output directory is already non-empty.

No individual action is silently skipped. This makes a successful command equivalent to a complete visual export.

## Verification

Focused tests will verify:

- A fixed seed selects exactly one sample per action.
- Repeating selection with the same seed returns identical absolute indices.
- The four target checkpoint jobs share the same selected index sequence.
- Source and target records contain the expected environment labels.
- Checkpoint and manifest paths resolve to the approved random-frame experiment locations.
- The single-sample renderer creates a PNG with an index-bearing filename and no PDF.
- Existing `eval.py --pose-viz` behavior remains unchanged.

A lightweight integration test will use a synthetic metadata fixture and mock model predictions. Full checkpoint inference is not required in the local test environment.

## Success Criteria

The feature is complete when one Linux command:

- Produces exactly one source-domain PNG per `env1_test` action.
- Produces exactly one PNG per `env2_test` action for each of the four target-data scales.
- Uses identical target dataset indices across the four finetuned models.
- Writes auditable source and target sample-selection CSVs.
- Produces no PDF files.
- Leaves all existing experiment artifacts unchanged.
