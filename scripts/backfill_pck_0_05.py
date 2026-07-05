from __future__ import annotations

"""Backfill ``pck_0_05`` into an existing experiment registry by re-running
evaluation (no training) on every completed checkpoint.

Usage::

    python scripts/backfill_pck_0_05.py \\
        --registry outputs/final_report_seed42_v4/experiment_registry.csv \\
        --dataset-root data/mmfi_pose \\
        --output outputs/final_report_seed42_v4/experiment_registry_pck05.csv
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_EVAL_MANIFEST_KEYS: dict[str, tuple[str, ...]] = {
    "source": ("env1_val", "env1_test"),
    "finetune_540": ("env2_val", "env2_test"),
    "finetune_scale": ("env2_test",),
}

_KEY_TO_PREFIX: dict[str, str] = {
    "env1_val": "val",
    "env1_test": "test",
    "env2_val": "val",
    "env2_test": "test",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill pck_0_05 into an experiment registry.",
    )
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_manifest_path(command: str) -> str:
    """Pull the ``--split-manifest`` value from a JSON-encoded argv list."""
    argv = json.loads(command)
    try:
        idx = argv.index("--split-manifest")
    except ValueError:
        raise ValueError("--split-manifest not found in command")
    return str(argv[idx + 1])


def _run_eval(
    dataset_root: Path,
    checkpoint: Path,
    manifest_path: str,
    manifest_key: str,
    output_dir: Path,
    batch_size: int,
    device: str,
) -> None:
    cmd = (
        sys.executable,
        str(ROOT / "eval.py"),
        "--dataset-root", str(dataset_root),
        "--checkpoint", str(checkpoint),
        "--split-manifest", manifest_path,
        "--manifest-key", manifest_key,
        "--output-dir", str(output_dir),
        "--batch-size", str(batch_size),
        "--device", device,
    )
    print(f"RUN: {subprocess.list2cmdline(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"eval.py failed with exit code {result.returncode}")


def _read_pck05(output_dir: Path) -> str:
    summary = output_dir / "benchmark_summary.csv"
    if not summary.is_file():
        raise FileNotFoundError(f"Missing benchmark_summary: {summary}")
    with summary.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    return row["pck_0_05"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    with args.registry.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        existing_fields = reader.fieldnames or []
        rows: list[dict[str, str]] = list(reader)

    new_val_field = "val_pck_0_05"
    new_test_field = "test_pck_0_05"
    for field in (new_val_field, new_test_field):
        if field not in existing_fields:
            existing_fields = list(existing_fields) + [field]
        for row in rows:
            row.setdefault(field, "")

    for row in rows:
        if row.get("status") != "completed":
            continue
        phase = row.get("phase", "")
        if phase not in _EVAL_MANIFEST_KEYS:
            continue

        checkpoint = Path(row["checkpoint_path"])
        if not checkpoint.is_file():
            print(f"SKIP {row['experiment_id']}: checkpoint not found", file=sys.stderr)
            continue

        try:
            manifest_path = _extract_manifest_path(row["command"])
        except Exception as exc:
            print(f"SKIP {row['experiment_id']}: {exc}", file=sys.stderr)
            continue

        eval_root = checkpoint.parent / "evaluations_pck05"
        for key in _EVAL_MANIFEST_KEYS[phase]:
            prefix = _KEY_TO_PREFIX[key]
            out_dir = eval_root / key
            try:
                if args.dry_run:
                    print(f"DRY-RUN {row['experiment_id']} {key} → {prefix}_pck_0_05")
                    continue
                _run_eval(
                    args.dataset_root,
                    checkpoint,
                    manifest_path,
                    key,
                    out_dir,
                    args.batch_size,
                    args.device,
                )
                value = _read_pck05(out_dir)
                field = f"{prefix}_pck_0_05"
                row[field] = value
                print(f"  {row['experiment_id']} {field}={value}")
            except Exception as exc:
                print(
                    f"ERROR {row['experiment_id']} {key}: {exc}",
                    file=sys.stderr,
                )

    # Write updated registry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=existing_fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
