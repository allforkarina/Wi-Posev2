# Handoff Guide

## Data and model contracts

The memmap dataset stores CSI as `(B, H, W, C)` and the collate function
permutes it to model input `(B, C, H, W)`. Pose targets and predictions use
OpenPose18 coordinates with shape `(B, 18, 2)`.

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
- `experiments/report_suite.py`: matched 30-task report matrix.

## Output rules

Use `outputs/` for checkpoints and generated artifacts. Do not commit raw
datasets, checkpoints, videos, images, or ad-hoc reports. The only retained
result evidence is `results/final_report_seed42_v6/`.

The historical tests and stale feature-visualization/diagnostic modules were
removed during the 2026-07-16 cleanup. Add focused synthetic tests before
changing a retained interface.
