# Script Catalogue

## Data preparation

- `scripts/data/audit_raw_ground_truth.py`: audit raw 17-joint GT,
  CSI/GT frame alignment, and the current 17-to-18 mapping without modifying
  the dataset. Run this before rebuilding delivery memmaps.
- `scripts/data/build_memmap.py`: build an interrupt-resumable CSI/pose memmap
  from raw MM-Fi CSI and audited flat GT arrays. Raw coordinates are preserved.
- `scripts/data/build_groundtruth.py`: merge pre-annotated GT arrays.
- `scripts/data/build_split_manifests.py`: create deterministic protocol
  manifests.
- `scripts/data/visualize_gt.py`: inspect one custom 18-joint GT frame.

## Evaluation

- `scripts/evaluation/benchmark_wipose.py`: evaluate one checkpoint on a
  manifest split and record accuracy, operations, latency, and memory.
- `scripts/evaluation/export_mechanism_viz.py`: export compact axial and
  joint-query attention evidence for a representative full checkpoint.

## Experiments

- `scripts/experiments/run_report_experiments.py`: run or resume the matched
  single-seed delivery matrix.
- `scripts/experiments/run_delivery_experiments.sh`: audit, rebuild data, run
  all three seeds serially, and summarize the complete delivery suite.
- `scripts/experiments/summarize_delivery_results.py`: aggregate multi-seed
  metrics and paired ablation effects.

## Media

- `scripts/media/export_demo_video.py`: export GT-vs-pred MP4/GIF video;
  FFmpeg must be available on `PATH`.
