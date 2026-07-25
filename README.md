# Wi-Posev2

Wi-Posev2 estimates a project-specific 18-joint 2D pose from Wi-Fi CSI. The repository keeps
the core training/evaluation path, deterministic experiment manifests, report
reproduction commands, benchmarking, and demo-video export.

## Setup

Create a Python 3.10 virtual environment and install the pinned dependencies
from `requirements-delivery.txt` (or use `environment.yml` with Conda). On the delivery server, use `.venv/bin/python`
directly so non-interactive jobs do not depend on shell activation.

## Core workflow

The complete single-GPU delivery suite is launched by
`scripts/experiments/run_delivery_experiments.sh`. It audits raw GT, builds an
interrupt-resumable memmap, runs all experiments serially for seeds 42, 123,
and 3407, and aggregates paired ablation effects. See
`docs/EXPERIMENTS.md` for the one-line server command and output contract.

`train.py` supports source-only training and cross-domain few-shot finetuning.
`eval.py` writes overall, tail-error, alignment, bone, joint-group, action,
environment, and temporal diagnostic CSVs. It can generate bounded pose
comparisons with `--pose-viz`. Dataset roots, checkpoints, and generated outputs are local
artifacts and must not be committed.

See [docs/HANDOFF.md](docs/HANDOFF.md) for interface contracts,
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for retained report evidence, and
[scripts/README.md](scripts/README.md) for the CLI catalogue.
