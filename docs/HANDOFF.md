# Handoff Guide

## Data and model contracts

The memmap dataset stores CSI as `(B, H, W, C)` and the collate function
permutes it to model input `(B, C, H, W)`. Pose targets and predictions use
the project's custom 18-joint coordinates with shape `(B, 18, 2)`.
`data/pose_schema.py` is the only authoritative source for the audited
Human3.6M17-to-project18 mapping, joint names, groups, torso diagonals, and
bone edges. The first two raw GT channels are preserved without resolution
assumptions, clipping, or `(0, 0)` invalid-point semantics.

`data/split_manifest.py` writes deterministic random-frame and temporal-block
splits. A manifest includes a hash of dataset metadata and source-training
normalization. Training records the resolved manifest provenance in checkpoint
`train_config`; evaluation validates the hash and rebuilds the model from that
same configuration before loading weights.

## Core modules

- `train.py`: training, finetuning, loss composition, metrics, checkpoints,
  and CSV learning curves.
- `eval.py`: checkpoint reconstruction, metric CSV output, and optional pose
  visualizations.
- `models/`: CSI feature processing, spatial and axial encoders, decoders,
  skeleton priors, input calibration, and wrist refinement.
- `experiments/report_suite.py`: matched delivery matrix with duplicate
  architectures trained once.

## Output rules

Use `outputs/` for checkpoints and generated artifacts. Do not commit raw
datasets, checkpoints, videos, images, or ad-hoc reports. The only retained
result evidence is `results/final_report_seed42_v6/`.

The historical tests and stale diagnostics were removed during cleanup.
Only focused synthetic contract tests for retained delivery workflows belong
in this handoff.
