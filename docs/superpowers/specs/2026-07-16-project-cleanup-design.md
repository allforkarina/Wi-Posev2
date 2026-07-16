# Wi-Posev2 Project Cleanup Design

## Goal

Produce a small, handoff-ready `main` branch that preserves the WiFlow
training, evaluation, preprocessing, experiment-reproduction, benchmarking,
and demo-video workflows. Remove stale tests, diagnostics, presentation
artifacts, temporary files, and redundant experiment outputs.

## Branch Strategy

1. Preserve the current source state in
   `codex/final-uncleaned-20260716`.
2. Perform all cleanup in `codex/project-cleanup-20260716`.
3. Do not merge the four commits unique to the former `main` branch.
4. After verification, force-update remote `main` with the cleanup commit
   using `--force-with-lease`.

The archive branch contains the deleted `CLAUDE.md` state and the two
untracked demo-video design documents. It intentionally does not contain
local videos, ZIP files, report images, `outputs/`, or `.superpowers/`.

## Retained Runtime Modules

Keep the following modules without broad refactoring:

- Root entry points: `train.py`, `eval.py`, and `dataloader.py`.
- Data modules: `data/memmap_dataset.py`, `data/split_manifest.py`, and
  `data/heatmap_gt.py`.
- All files in `models/`, because the report suite uses the axial-mode,
  decoder, CSI-calibration, skeleton-prior, and wrist-refinement variants.
- Evaluation modules: `evaluation/benchmark.py` and
  `evaluation/pose_viz.py`.
- Experiment module: `experiments/report_suite.py`.

`train.py` remains the training module interface. `eval.py` remains the
metrics and qualitative-pose interface, but no longer exposes stale feature
visualization options. No large module split is part of this cleanup.

## Script Layout

Move each retained standalone CLI into a responsibility-based directory.

```text
scripts/
  README.md
  data/
    build_memmap.py
    build_groundtruth.py
    build_split_manifests.py
    visualize_gt.py
  evaluation/
    benchmark_wipose.py
  experiments/
    run_report_experiments.py
  media/
    export_demo_video.py
```

Update the `run_report_experiments.py` implementation to invoke the moved
manifest-builder and benchmark scripts. Update all user-facing documentation
to use the new paths. Make the raw-source and output paths of
`build_memmap.py` required CLI inputs instead of machine-specific defaults.

## Removed Source Modules

Remove the following modules because they are obsolete, empty, diagnostic-only,
presentation-only, or one-shot migration code:

- `pose_targets.py`
- `evaluation/feature_viz.py`
- `evaluation/hooks.py`
- `scripts/backfill_pck_0_05.py`
- `scripts/stem_feature_diagnostic.py`
- `scripts/export_report_pose_visualizations.py`
- `tests/`

Update `evaluation/__init__.py` and `eval.py` so no import or CLI option
references the removed feature-visualization modules.

## Removed Documentation and Local Artifacts

Remove historical development plans and obsolete project notes:

- `docs/superpowers/`
- `docs/memmap_migration_plan.md`
- `REASONIX.md`
- tracked preview files in `data/gt_merged/`

Remove all local-only artifacts and configure `.gitignore` to prevent their
return:

- `.agents/`, `.claude/`, `.superpowers/`
- caches, bytecode, and pytest temporary directories
- demo-video directories and ZIP files
- report image directories and ZIP files
- `outputs/`

No Word document is currently present; the existing `*.docx` ignore rule
remains.

## Final Report Evidence

Retain only the final 2026-07-05 v6 report evidence under
`results/final_report_seed42_v6/`:

- `experiment_registry.csv`
- random-frame and temporal-block16 `.npz` manifests and sidecar JSON files
- every retained experiment's `train_log.csv`
- test-set `benchmark_summary.csv`, `per_action_metrics.csv`,
  `per_environment_metrics.csv`, and `per_joint_metrics.csv`
- benchmark `runtime_metrics.csv`

Delete the v4 result batch, all PNG/PDF visualizations, all ZIP files, all
demo videos, pose-comparison report images, and redundant validation or
duplicate benchmark/evaluation artifacts.

The retained results must support the following matched comparisons:

| IDs | Claim | Controlled comparison and evidence |
| --- | --- | --- |
| A1-A4 | Axial attention order changes CSI cue selection. | Same split, seed, decoder, and bone loss; compare test MPJPE/PCK and action/joint rows. |
| D1/D3 | Decoder structure changes pose extraction. | Compare MLP and hierarchical decoders with A1. |
| B1 | Skeleton supervision changes structural plausibility. | Compare zero bone-loss weight with A1 using bone error and joint metrics. |
| F1-F5 | Adaptation depth changes few-shot transfer. | Fixed 540-shot protocol; vary only trainable group. |
| V2-V4 | More target samples change adaptation quality. | Use the selected trainable group and vary only few-shot size. |
| Both splits | Conclusions are split-sensitive. | Run the same matrix for random-frame and temporal-block16 manifests. |

`docs/EXPERIMENTS.md` will describe these comparisons and state that v6 is a
single-seed final run, not a multi-seed statistical claim.

## Handoff Documentation

Create these handoff artifacts:

- `README.md`: purpose, setup, dataset contract, shortest training/evaluation
  commands, and directory guide.
- `environment.yml`: Python and runtime dependencies required by the retained
  workflows.
- `docs/HANDOFF.md`: CSI/pose tensor shapes, checkpoints, manifests, module
  interfaces, output rules, and removed functionality.
- `docs/EXPERIMENTS.md`: report matrix, v6 evidence location, controls, and
  limitations.
- `scripts/README.md`: script categories, inputs, outputs, and commands.
- `AGENTS.md`: concise maintenance rules plus manifest-aware demo-video
  invocation and FFmpeg requirements.

## Verification

Because the handoff branch intentionally removes `tests/`, verify the cleaned
repository with:

1. Python `compileall` on all retained modules.
2. Import checks for `train`, `eval`, `dataloader`, `data`, `models`,
   `evaluation`, and `experiments`.
3. `--help` checks for every retained CLI.
4. A static check that the experiment runner names the moved script paths.
5. Git and filesystem checks confirming removed tests, stale modules, caches,
   videos, ZIP files, v4 results, and large visualization outputs are absent.

The pre-cleanup baseline is `134 passed, 4 warnings`; it documents the source
state before the test suite is intentionally removed.
