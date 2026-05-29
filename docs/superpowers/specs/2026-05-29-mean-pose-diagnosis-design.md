# Mean-Pose Collapse Diagnosis (v1)

## Problem

Cross-domain fine-tuned model may predict a "mean pose" — approximately the same
output regardless of input — exploiting the fact that average pose achieves
decent MPJPE/PCK when the target distribution is not too diverse. This breaks
per-sample accuracy while global metrics look fine.

## Solution

Add lightweight diagnostic metrics to `eval.py` computed during the existing
single-pass evaluation loop. No extra inference cost.

## Diagnostic Metrics

| Metric | Computation | Mean-pose signal |
|--------|------------|-----------------|
| `pred_var` | Per-joint variance of predictions across all eval samples | Near zero |
| `gt_var` | Per-joint variance of GT across all eval samples | Reference |
| `var_ratio` | `pred_var / gt_var` | < 0.3 strongly suggests collapse |
| `mean_pose_dist` | L2 distance between mean prediction and mean GT (per-joint) | Small but `var_ratio` also small = model outputs a single pose regardless of input |

All metrics are computed per-joint (18 joints, 2 coordinates each). Variance is
over the sample axis, then averaged over x/y coordinates.

## Output

**Terminal**: Prints diagnostic summary after existing metrics.

```
--- Diagnostic Metrics ---
  overall_pred_var:         0.1234
  overall_gt_var:           0.4567
  overall_var_ratio:        0.2702
  overall_mean_pose_dist:   0.0890
```

**CSV**: `per_joint_diagnostic.csv` (18 rows, one per joint):
```
joint_index,pred_var,gt_var,var_ratio,mean_pose_dist
0,0.0123,0.0456,0.270,0.0089
...
```

## Code Changes

1. **`eval.py`** — `run_evaluation`:
   - Collect `all_predictions` and `all_targets` lists alongside existing accumulators.
   - After the loop, compute diagnostics via `_compute_diagnostics()`.
   - Print diagnostics to terminal.
   - Write `per_joint_diagnostic.csv` via existing `_write_csv`.

2. **`eval.py`** — Add `_compute_diagnostics(all_preds, all_targets) -> dict`:
   - Computes per-joint prediction variance (`pred_var`), GT variance (`gt_var`),
     ratio (`var_ratio`), and L2 distance between per-joint means (`mean_pose_dist`).
   - Returns an aggregated `overall_*` dict and a per-joint list of dicts.

## Dependencies

- `numpy` — already imported in eval.py.
- No new imports needed.

## Testing

- Run `eval.py` on an existing checkpoint with `--eval-envs`, verify diagnostic
  metrics print to terminal and `per_joint_diagnostic.csv` is written.
- Manually inspect: a good model should have `var_ratio > 0.5`; a collapsed model
  should show `var_ratio < 0.3`.
