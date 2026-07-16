# Wi-Posev2

Wi-Posev2 estimates OpenPose18 2D poses from Wi-Fi CSI. The repository keeps
the core training/evaluation path, deterministic experiment manifests, report
reproduction commands, benchmarking, and demo-video export.

## Setup

```bash
conda env create -f environment.yml
conda activate WiFiPose
```

## Core workflow

```bash
python scripts/data/build_memmap.py --src <raw-mmfi-root> --dst <memmap-root>
python scripts/data/build_split_manifests.py --dataset-root <memmap-root> --output-dir outputs/manifests
python train.py --mode source_only --dataset-root <memmap-root> --output-dir outputs/source
python eval.py --dataset-root <memmap-root> --checkpoint outputs/source/best_val_pck_0_2.pth --output-dir outputs/eval
python scripts/media/export_demo_video.py --help
```

`train.py` supports source-only training and cross-domain few-shot finetuning.
`eval.py` writes metric CSVs and can generate pose comparisons with
`--pose-viz`. Dataset roots, checkpoints, and generated outputs are local
artifacts and must not be committed.

See [docs/HANDOFF.md](docs/HANDOFF.md) for interface contracts,
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for retained report evidence, and
[scripts/README.md](scripts/README.md) for the CLI catalogue.
