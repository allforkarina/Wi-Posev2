# Script Catalogue

## Data preparation

- `scripts/data/audit_raw_ground_truth.py`: audit raw 17-joint GT,
  CSI/GT frame alignment, and the current 17-to-18 mapping without modifying
  the dataset. Run this before rebuilding delivery memmaps.
- `scripts/data/build_memmap.py`: build CSI and pose memmap files from raw
  MM-Fi data; requires `--src` and `--dst`.
- `scripts/data/build_groundtruth.py`: merge pre-annotated GT arrays.
- `scripts/data/build_split_manifests.py`: create deterministic protocol
  manifests.
- `scripts/data/visualize_gt.py`: inspect one OpenPose18 ground-truth frame.

## Evaluation

- `scripts/evaluation/benchmark_wipose.py`: evaluate one checkpoint on a
  manifest split and record accuracy, operations, latency, and memory.

## Experiments

- `scripts/experiments/run_report_experiments.py`: run or resume the matched
  final-report matrix.

## Media

- `scripts/media/export_demo_video.py`: export GT-vs-pred MP4/GIF video;
  FFmpeg must be available on `PATH`.
