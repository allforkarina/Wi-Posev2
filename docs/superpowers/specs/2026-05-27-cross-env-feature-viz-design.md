# Cross-Environment Feature Difference Visualization — Design Spec

> Date: 2026-05-27 | Status: draft

## Motivation

Current `feature_viz.py` generates per-sample figures but provides no cross-environment
comparison.  To identify how environmental conditions affect CSI features relevant to
human pose, we need visualizations that directly compare intermediate features between
two domains (e.g., lab vs corridor).

## Scope

- Two environments only (source vs target), specified via CLI.
- Comparisons at the `axial_encoder` output `[B, 256, 29, 16]` (main focus) and
  spatial/temporal attention weights (secondary focus).
- Both global aggregation and per-action breakdowns.

## Architecture

New module: `evaluation/cross_env_viz.py`.  Invoked from `eval.py` via a new
`--cross-env-viz` flag.  Reuses the hook system (`evaluation/hooks.py`) and
stratified sampling (`_collect_action_env_samples` from `feature_viz.py`).

```
eval.py
  ├── parse_args()  ← add --cross-env-viz, --source-env, --target-env
  └── main()
        └── if cross_env_viz:
              cross_env_viz.run_cross_env_visualization(model, loader, ...)

evaluation/cross_env_viz.py
  ├── collect_env_features()        → {env: {action: [features], ...}}
  ├── figA_channel_activation_delta()
  ├── figB_correlation_delta()
  ├── figC_attention_offset()
  ├── figD_feature_distribution_shift()
  └── run_cross_env_visualization()  ← orchestrator
```

## Figures

### A — Channel Activation Delta

Global: mirrored horizontal bar chart (source channels right, target channels left),
with a zero-axis separator at centre.  Each bar = one of 256 channels, sorted by mean
activation.  Below: a vertical delta heatmap `[256, 1]` using RdBu_r colormap with
symmetric vmin/vmax.

Per-action: small-multiples sparkline grid, one row per action in the dataset,
showing per-channel Δ (source − target) as a horizontal inline bar.

### B — Correlation Delta (Δr)

Global: 6×3 grid, each cell is a `[29, 16]` heatmap of `r(source) - r(target)`.
Warm color (#C46A4A) = source correlation stronger; cool (#4F7EBF) = target
correlation stronger.  Zero-centre is neutral grey (#FCFCFC).  White ✕ marks max
|Δr|.  Anatomical group borders (head/upper/lower) as dashed rectangles.

Per-action: select top 3 actions by mean |Δr|, show as 3-column side-by-side grids
(one column per action, each column = 18-joint mini heatmap column).

### C — Attention Offset

3×2 grid.  Rows: source, target, delta.  Columns: spatial attention, temporal
attention.  Delta uses RdBu_r with symmetric vmin/vmax.  One colorbar per row.

### D — Feature Distribution Shift

One panel: PCA 2D of axial_encoder global-average-pooled features `[N, 256]`,
colored by environment.  68% confidence ellipses per environment.  Maximum Mean
Discrepancy (MMD) with RBF kernel annotated in upper-right corner.

## Aesthetic Specification (Editorial Scientific)

All figures follow the same style constants as the updated `feature_viz.py`:

| Element | Value |
|---------|-------|
| Anatomy colors | head=#C46A4A, upper=#5B55A1, trunk=#3B8A6A, lower=#4F7EBF |
| Figure facecolor | #FAFAFA (warm grey) |
| Axes facecolor | #FCFCFC |
| Axes edgecolor | #CCCCCC |
| Grid | alpha=0.25, color=#D0D0D0, linestyle=--, linewidth=0.4 |
| Suptitle | fontsize=13, color=#333333, fontweight=bold |
| Colorbar | fraction=0.038, no outline, tick labelsize=7 |
| Spine visibility | top/right spines hidden |
| Legend | framealpha=0.85, edgecolor=#DDDDDD |

## CLI Additions

```
python eval.py --dataset-root data/mmfi_pose \
    --checkpoint outputs/train/best_val_mpjpe.pth \
    --output-dir outputs/eval \
    --cross-env-viz \
    --source-env lab \
    --target-env corridor
```

New arguments:
- `--cross-env-viz` (store_true): enable cross-environment visualization.
- `--source-env` (str, default="lab"): source environment name.
- `--target-env` (str, default="corridor"): target environment name.

## Data Flow

```
test DataLoader
  → wiflow_hooks(model, ["axial_encoder", "axial_encoder.spatial_attention",
                           "axial_encoder.temporal_attention"])
  → for each batch: collect features + attention + metadata (action, env)
  → filter to {source_env, target_env} only
  → group by (env, action)
  → per-action: compute channel means, per-spatial-position correlations, attention avg
  → generate figs A–D
```

## Dependencies

- matplotlib (existing)
- numpy (existing)
- torch (existing)
- scipy.stats.pearsonr (existing)
- sklearn.decomposition.PCA (existing)
- evaluation.hooks (existing)
- feature_viz._ANATOMY_COLORS, _ANATOMY_GROUPS, _JOINT_NAMES, _add_colorbar,
  _apply_spacing, _save_fig (import from feature_viz for consistency)

## Not in Scope

- More than 2 environments
- Real-time / interactive visualization
- CECE feature comparison (focus is on encoder output regardless of CECE state)
- Per-subject breakdown
