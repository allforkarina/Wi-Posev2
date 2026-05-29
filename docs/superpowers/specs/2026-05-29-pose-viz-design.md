# Pose Joint Scatter Visualization Design

## Overview

Add a `--pose-viz` flag to `eval.py` that generates per-sample joint scatter plots comparing GT and predicted OpenPose18 keypoints. Each joint gets a distinct color; no skeleton bones are drawn. This feature is independent from the existing `--feature-viz` pipeline.

## Trigger

- `eval.py` gains a new `--pose-viz` flag (boolean, default `False`).
- When `--pose-viz` is set, the scatter plot pipeline runs **after** the standard evaluation metrics pass but **instead of** `--feature-viz`. The two flags are mutually exclusive — if both are set, the script exits with an error.
- Reuses the same `--num-action-samples`, `--output-format`, `--figure-width`, `--figure-height` CLI args already defined for `--feature-viz`.

## Sampling

- Action × environment stratified, identical to `feature_viz._collect_action_env_samples`.
- Default: `num_action_samples=3` samples per action, each from a distinct environment.
- Samples are extracted during the evaluation pass (reuse `loader` + `prepare_model_input` + `extract_prediction_keypoints`).

## Plot Specification (Clean Modern style)

Per sample, one figure (`figsize=(8, 8)`):

| Element | Style |
|---------|-------|
| Background | White `#ffffff` with subtle gray grid (`#e8e8e8`, 0.5pt) |
| GT keypoints | Dashed hollow circles, same color as the joint, 1.8pt ring |
| Prediction keypoints | Filled circles with dark `#333` 1pt border, radius 6.5 |
| Error vectors | Thin gray (`#ccc`) dashed lines from GT → Prediction |
| Joint index labels | `fontsize=8, fontweight='bold'`, offset (+8, -8) from prediction point |
| Legend | 2 entries: GT (dashed ring marker) and Prediction (filled circle marker) |
| Axes | `"Normalized X"` / `"Normalized Y"`, `aspect='equal'`, Y-axis inverted |
| Title | `"Joint Prediction vs GT — {action} / {environment}"`, bold 12pt |
| Margins | 10% padding from data min/max |

### Color Palette (18 vibrant categorical, one per joint index)

```
#E6194B, #3CB44B, #FFE119, #4363D8, #F58231, #911EB4, #46F0F0, #F032E6,
#BCF60C, #FABEBE, #008080, #E6BEFF, #9A6324, #FFFAC8, #800000, #AAFFC3,
#808000, #FFD8B1
```

Each joint gets exactly one color. Joint index `j` uses `palette[j]`.

## Output Layout

```
{output_dir}/
  pose_viz/
    {action}_{env}_s{idx}/
      fig_pose_scatter.png   (or .pdf)
```

No overview composite or per-action composite is generated.

## Code Changes

1. **Strip from `evaluation/feature_viz.py`**:
   - Remove `_fig0_joint_scatter` function (~50 lines).
   - Remove the `_fig0_joint_scatter(...)` call from `run_feature_visualization` ~line 1077.
   - Remove the `_fig0_joint_scatter` re-generation in `_build_action_composites` ~lines 916–922.

2. **Add `evaluation/pose_viz.py`**:
   - Contains `run_pose_visualization(model, loader, device, output_dir, num_action_samples, output_format, figure_width, figure_height)`.
   - Reuses `_collect_action_env_samples` from `feature_viz.py` (move to shared utils or import directly).
   - Per-sample: draw and save one figure using matplotlib.

3. **Modify `eval.py`**:
   - Add `--pose-viz` argument.
   - Mutual exclusion check with `--feature-viz`.
   - Wire the new flag to `run_pose_visualization`.

## Dependencies

- `matplotlib`, `numpy`, `torch` — already available.
- Reuses `feature_viz._collect_action_env_samples`, `feature_viz._flatten_samples`, `feature_viz._save_fig`, `feature_viz._apply_spacing`.
- Reuses `train.prepare_model_input`, `train.extract_prediction_keypoints`.

## Testing

- Manual verification: run `eval.py --pose-viz` on an existing checkpoint, confirm PNG files appear under `output_dir/pose_viz/`, visually inspect one scatter plot.
- No new automated tests (visualization output is qualitative).
