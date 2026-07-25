#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=""
RAW_DATASET_ROOT=""
GROUND_TRUTH_ROOT=""
DATASET_ROOT=""
OUTPUT_ROOT=""
PYTHON_BIN=""
SOURCE_EPOCHS=50
FINETUNE_EPOCHS=30
BATCH_SIZE=64
WORKERS=4

usage() {
  echo "Usage: $0 --project-root PATH --raw-dataset-root PATH --ground-truth-root PATH --dataset-root PATH --output-root PATH [--python PATH] [--source-epochs N] [--finetune-epochs N] [--batch-size N] [--workers N]"
}

while (($#)); do
  case "$1" in
    --project-root) PROJECT_ROOT="$2"; shift 2 ;;
    --raw-dataset-root) RAW_DATASET_ROOT="$2"; shift 2 ;;
    --ground-truth-root) GROUND_TRUTH_ROOT="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --source-epochs) SOURCE_EPOCHS="$2"; shift 2 ;;
    --finetune-epochs) FINETUNE_EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in PROJECT_ROOT RAW_DATASET_ROOT GROUND_TRUTH_ROOT DATASET_ROOT OUTPUT_ROOT; do
  if [[ -z "${!required}" ]]; then
    echo "Missing required option for $required" >&2
    usage >&2
    exit 2
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

on_interrupt() {
  echo "Interrupted safely. Re-run the same command; completed stages and experiments will be skipped." >&2
  exit 130
}
trap on_interrupt INT TERM

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT"

"$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(), 'CUDA is required'; print(torch.__version__, torch.cuda.get_device_name(0))"

AUDIT_DIR="$OUTPUT_ROOT/gt_audit"
if [[ ! -f "$AUDIT_DIR/audit_summary.json" ]]; then
  "$PYTHON_BIN" scripts/data/audit_raw_ground_truth.py --raw-dataset-root "$RAW_DATASET_ROOT" --ground-truth-root "$GROUND_TRUTH_ROOT" --output-dir "$AUDIT_DIR" --preview-file-count 8
fi

"$PYTHON_BIN" scripts/data/build_memmap.py --src "$RAW_DATASET_ROOT" --gt-dir "$GROUND_TRUTH_ROOT" --dst "$DATASET_ROOT" --workers "$WORKERS" --resume

for SEED in 42 123 3407; do
  "$PYTHON_BIN" scripts/experiments/run_report_experiments.py --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT/seed$SEED" --split-modes random_frame temporal_block --seed "$SEED" --source-epochs "$SOURCE_EPOCHS" --finetune-epochs "$FINETUNE_EPOCHS" --batch-size "$BATCH_SIZE" --device cuda --resume
done

"$PYTHON_BIN" scripts/experiments/summarize_delivery_results.py --output-root "$OUTPUT_ROOT"
echo "Delivery experiment suite completed: $OUTPUT_ROOT"
