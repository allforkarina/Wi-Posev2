# Pose Joint Scatter Visualization Design (v2)

## Overview

Add a `--pose-viz` flag to `eval.py` that generates per-subject joint scatter plots comparing GT and predicted OpenPose18 keypoints. Each joint gets a distinct color; no skeleton bones are drawn. This feature is independent from the existing `--feature-viz` pipeline.

## Trigger

- `eval.py` gains a new `--pose-viz` flag (boolean, default `False`).
- When `--pose-viz` is set, the scatter plot pipeline runs **after** the standard evaluation metrics pass but **instead of** `--feature-viz`. The two flags are mutually exclusive — if both are set, the script exits with an error.
- Reuses `--output-format`, `--figure-width`, `--figure-height` CLI args already defined for `--feature-viz`.
- Does **not** use `--num-action-samples` (covers all subjects; see Sampling).

## Sampling

**Strategy**: For every (action, subject) pair in the eval-filtered dataset, select exactly **one middle frame**.

- Dataset structure: 27 actions × 40 subjects × 297 frames. Subjects 0–9 are env1, 10–19 env2, 20–29 env3, 30–39 env4.
- Middle frame: index 148 (`297 // 2`) within each subject's contiguous frame range.
- The `MemmapDataset` loaded with `split="all"` and `envs=<eval-envs>` provides the filtered `self.indices`. The sampling logic locates, for each (action, subject), all rows whose meta matches, sorts them, and picks the 148th.
- **Fixed**: same frame every eval run. No randomness.

**Scope per eval**:
- 27 actions × 10 subjects per env = **~270 samples** per env.
- Example: `--eval-envs env2` → 270 scatter plots.

**Implementation approach**:
- Load `MemmapDataset` with the eval env filter, iterate its metadata once to build `{ (action, subject): absolute_frame_index }`, storing the 148th frame of each group.
- Build a `Subset` pointing to those frames for a single-epoch `DataLoader` batch-inference pass.

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
| Title | `"Joint Prediction vs GT — {action} / {subject} / {environment}"`, bold 12pt |
| Margins | 10% padding from data min/max |

### Color Palette (18 vibrant categorical, one per joint index)

```
#E6194B, #3CB44B, #FFE119, #4363D8, #F58231, #911EB4, #46F0F0, #F032E6,
#BCF60C, #FABEBE, #008080, #E6BEFF, #9A6324, #FFFAC8, #800000, #AAFFC3,
#808000, #FFD8B1
```

Joint index `j` uses `palette[j]`.

## Output Layout

```
{output_dir}/
  pose_viz/
    {action}_{env}_{subject}/
      fig_pose_scatter.png   (or .pdf)
```

No overview composite or per-action composite is generated. One directory per (action, subject) pair.

## Code Changes

1. **Strip from `evaluation/feature_viz.py`**:
   - Remove `_fig0_joint_scatter` function (~50 lines).
   - Remove the `_fig0_joint_scatter(...)` call from `run_feature_visualization` ~line 1077.
   - Remove the `_fig0_joint_scatter` re-generation in `_build_action_composites` ~lines 916–922.

2. **Add `evaluation/pose_viz.py`**:
   - `run_pose_visualization(model, dataset, device, output_dir, output_format, figure_width, figure_height, batch_size, num_workers)`.
   - Internally queries dataset metadata to locate middle frames for each (action, subject), builds a Subset DataLoader, runs one forward pass, and saves scatter plots.
   - Uses matplotlib with the Clean Modern style constants defined inline.

3. **Modify `eval.py`**:
   - Add `--pose-viz` argument.
   - Mutual exclusion check with `--feature-viz`.
   - Wire `--pose-viz` to pass the `MemmapDataset` (not the evaluation loader) to `run_pose_visualization`.

## Dependencies

- `matplotlib`, `numpy`, `torch` — already available.
- Reuses `feature_viz._save_fig` (or duplicates the save logic inline for self-containment).
- Reuses `train.prepare_model_input`, `train.extract_prediction_keypoints`.

## Testing

- Manual: run `eval.py --pose-viz --eval-envs env2` on an existing checkpoint, verify ~270 directories are created under `output_dir/pose_viz/`, visually inspect 2–3 scatter plots.
- No new automated tests (visualization output is qualitative).
