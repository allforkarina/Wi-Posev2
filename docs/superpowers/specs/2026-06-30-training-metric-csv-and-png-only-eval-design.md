# Training Metric CSV and PNG-Only Evaluation Design

## Scope

Add per-epoch metric CSV files to both source-only training and few-shot
finetuning, while retaining the existing `train_log.csv`. Remove PDF generation
from evaluation visualizations. Do not add per-joint training curves or change
the existing evaluation CSV schema.

## Training outputs

Each file contains one row per completed epoch and is appended in epoch order.
Source-only rows contain `train_*` and `val_*` columns. Finetuning rows contain
training columns only because the current finetuning workflow has no validation
loader.

- `loss.csv`: total, coordinate, and bone losses plus any enabled optional loss
  components, including latent structure, encoder relation, distal, wrist
  direction, target, and source replay losses.
- `mpjpe.csv`: MPJPE for each available split.
- `pck_0_1.csv`, `pck_0_2.csv`, `pck_0_3.csv`, `pck_0_4.csv`, and
  `pck_0_5.csv`: one PCK threshold per file for each available split.
- `bone_error.csv`: mean absolute error between predicted and ground-truth bone
  lengths over the existing OpenPose18 skeleton edges for each available split.

Every file includes an `epoch` column. Existing `train_log.csv` remains
available for compatibility. CSV files are written only after an epoch finishes,
using the existing append behavior so prior epoch rows are preserved.

## Metric computation

`run_epoch` and `run_finetune_epoch` return the full existing PCK threshold set,
MPJPE, and bone error in addition to loss values. Bone error uses the same
OpenPose18 topology and Euclidean bone lengths as the current bone loss, but is
reported as an explicit performance metric rather than only as a weighted
training objective.

No per-joint training metric is introduced. Detailed joint, action,
environment, and mean-pose-collapse analysis remains in the existing evaluation
outputs:

- `per_joint_metrics.csv`
- `per_action_metrics.csv`
- `per_environment_metrics.csv`
- `per_joint_diagnostic.csv`

## Evaluation visualization outputs

Remove the `--output-format` CLI argument from `eval.py`. Feature visualization
helpers always save `.png` files and no longer maintain PDF/both output-format
state. Pose visualization already saves PNG only and remains unchanged.
Existing PDF files in old output directories are not deleted.

## Compatibility and failure behavior

- Existing checkpoints and model reconstruction are unchanged.
- Existing `train_log.csv` schema and contents remain unchanged.
- Metric CSV headers are stable for one training run because optional loss
  configuration is fixed before epoch 1.
- A restarted run targeting a non-empty output directory continues the existing
  append semantics; automatic epoch deduplication is outside this change.

## Verification

- Unit-test row construction and file separation for source-only and finetune
  metrics.
- Unit-test bone error using synthetic poses with known bone-length differences.
- Unit-test that feature figure saving creates PNG and never PDF.
- Unit-test that the evaluation CLI no longer accepts `--output-format`.
- Run the complete test suite in the `WiFiPose` Conda environment.

## Documentation

Update `AGENTS.md` to document the new training artifacts and PNG-only feature
visualization behavior.
