# Eval Feature Visualization Implementation Plan

> **For agentic workers:** Execute tasks 1-6 sequentially after code review.

**Goal:** Refactor eval.py to remove skeleton connections, add 6 research-grade feature visualization figures with dynamic forward hooks, auto-detect decoder type for applicable visualizations.

**Architecture:** New `eval_hooks.py` encapsulates all hook registration/cleanup via contextmanager. Minimal model change (2 lines in `wiflow_axial_encoder.py`: `need_weights=False` → `True`). eval.py rewritten to support `--feature-viz` flag; old `save_visualizations` removed and replaced. Sampling: 3 representative samples per action type across different environments. Output: `outputs/eval/feature_viz/{sample_id}/` for per-sample figs 1-5, `outputs/eval/feature_viz/_global/` for fig 6.

**Model module paths (verified):**
- `spatial_encoder.antenna_mixer` — nn.Sequential (1×1 conv), output shape [B, 3, 64, 114]
- `spatial_encoder.feature_stem` — nn.Sequential, output shape [B, 32, 64, 114]
- `spatial_encoder.resblock1` — SymmetricResidualDownsampleBlock, output shape [B, 64, 32, 57]
- `spatial_encoder.resblock2` — SymmetricResidualDownsampleBlock, output shape [B, 128, 16, 29]
- `spatial_encoder.resblock3` — SymmetricResidualDownsampleBlock, output shape [B, 128, 16, 29]
- `axial_encoder.spatial_attention` — nn.MultiheadAttention, weights [B*T, 8, 29, 29]
- `axial_encoder.temporal_attention` — nn.MultiheadAttention, weights [B*29, 8, 16, 16]
- `axial_encoder` — entire module, output shape [B, 256, 29, 16]
- `decoder.cross_attention_layers.0` / `.1` / `.2` — WiFlowJointCrossAttentionLayer (joint decoder)
- `decoder.stages.0` / `.1` / `.2` — WiFlowHierarchicalJointDecoderStage (hierarchical decoder)
- `decoder.decoders.0` / `.1` / `.2` — WiFlowHeatmapDecoder (heatmap_msfn decoder)

---

## Figure Specifications Reference (from user design doc)

### Global Specs
- Canvas: `figsize=(18, H)`, width=18", height per panel count
- Output: 300 DPI, PDF + PNG
- Font: `DejaVu Sans`
- Colorbar: uniform width=0.15", match subplot height
- Font sizes: suptitle 14pt bold, subplot title 11pt, axis labels 10pt, tick labels 9pt, annotations 8pt, colorbar label 8pt
- Spacing: `subplots_adjust(hspace=0.45, wspace=0.35, left=0.07, right=0.93, top=0.92, bottom=0.06)`

### Fig 1: Antenna Channel Response Analysis
- Layout: `(18, 4)`, 1×7 (3 input cols + separator col + 3 output cols)
- Colormap: `RdBu_r`, vmin/vmax = 99th percentile of abs
- Cross-corr overlay on output panels: color `#E05C30`, lw=1pt
- Separator col: text "antenna\nmixer" with arrow, 11pt

### Fig 2: Symmetric Downsampling Trajectory
- Layout: `(18, 6)`, 2×4
- Row 1: PCA RGB on first 8 channels, bilinear resize to 224×224
- Row 2: Channel variance histogram, bins=32, alpha=0.75
- Colors: input `#378ADD`, block1 `#534AB7`, block2 `#1D9E75`, block3 `#D85A30`
- Mean line: dashed vertical + "mean=x.xx" annotation, 8pt

### Fig 3: Axial Attention Maps
- Layout: `(18, 8)`, 2×5 (row1=spatial, row2=temporal; cols: avg+4 individual heads)
- Colormap: `viridis`, vmin=0
- Diagonal: white dashed axline
- Entropy: horizontal bar on right of each panel (width=15%)

### Fig 4: Joint Query Trajectory
- Layout: `(18, 10)`, 2×(L+1) (row1=t-SNE, row2=cosine sim)
- t-SNE: perplexity=5, point size=80, white edge lw=0.5pt
- Colors: Head `#E05C30`, Upper `#534AB7`, Trunk `#1D9E75`, Lower `#378ADD`
- Cosine sim: `coolwarm`, vmin=-1, vmax=1
- Legend: on last column, outside subplot

### Fig 5a: PCM Radar
- Layout: `(10, 10)`, single polar plot
- 18 joints, 3 stages: dashed/dotdash/solid; colors `#B5D4F4`/`#378ADD`/`#0C447C`
- Alpha fill=0.1

### Fig 5b: PAF Direction Consistency
- Layout: `(8, 8)`, single subplot
- Static skeleton background (`#CCCCCC`, lw=1pt), bg `#F5F5F5`
- Overlay circles (s=120) colored by cosine sim, colormap `RdYlGn`, vmin=0, vmax=1

### Fig 6: Feature-Pose Correlation Landscape
- Layout: `(18, 12)`, 6×3
- 18 panels, `coolwarm`, vmin=-1, vmax=1
- Shared colorbar: right side, height=full, width=0.2"
- White cross × at max |r| position
- Anatomical group borders (dashed, 1pt):
  - Head+Trunk (top-left 6), Upper (middle 6), Lower (bottom-right 6)
  - Colors match Fig 4 groups

### Overview Composite
- Layout: `(18, 12)`, 2×3 thumbnails
- Per thumbnail: width=6", fig label (Fig 1–6) 12pt bold at top-left

---

## Tasks

### Task 1: Minimal model change — enable attention weights

**File:** `models/wiflow_axial_encoder.py`

- [ ] **Step 1:** Change `need_weights=False` → `need_weights=True` in `_apply_spatial_attention` (line ~85)
- [ ] **Step 2:** Same change in `_apply_temporal_attention` (line ~97)

### Task 2: Create `eval_hooks.py`

**File:** `eval_hooks.py` (new)

- [ ] **Step 1:** Create file with `WiFlowHookContext` class and `wiflow_hooks` contextmanager

### Task 3: Refactor `eval.py`

**File:** `eval.py`

- [ ] **Step 1:** Remove `_draw_skeleton`, `_sanitize_filename`, `save_visualizations` functions
- [ ] **Step 2:** Remove `OPENPOSE_BONE_EDGES`, `Axes` from imports
- [ ] **Step 3:** Replace `parse_args()`: add `--feature-viz`, `--num-action-samples`; remove `--max-visualizations`
- [ ] **Step 4:** Replace `main()`: conditional feature viz call

### Task 4: Create `eval_feature_viz.py`

**File:** `eval_feature_viz.py` (new)

- [ ] **Step 1:** Sampling helpers: `_collect_action_env_samples`, `_flatten_samples`
- [ ] **Step 2:** Fig 1: Antenna Channel Response Analysis (1×7 layout)
- [ ] **Step 3:** Fig 2: Downsampling Trajectory (2×4 with PCA + variance)
- [ ] **Step 4:** Fig 3: Axial Attention Maps (2×5 with avg + individual heads)
- [ ] **Step 5:** Fig 4: Joint Query Trajectory (2×(L+1) with t-SNE + cosine sim)
- [ ] **Step 6:** Fig 5a: PCM Radar (polar spider chart)
- [ ] **Step 7:** Fig 5b: PAF Direction Consistency (skeleton overlay)
- [ ] **Step 8:** Fig 6: Feature-Pose Correlation Landscape (6×3 global)
- [ ] **Step 9:** Overview composite figure (2×3 thumbnails)
- [ ] **Step 10:** Orchestrator: `run_feature_visualization`

### Task 5: Integration test

- [ ] **Step 1:** Verify eval without `--feature-viz` works
- [ ] **Step 2:** Verify eval with `--feature-viz` generates all figures
- [ ] **Step 3:** Verify output directory structure

### Task 6: Code review self-check

- [ ] No skeleton logic remains in eval.py
- [ ] `--feature-viz` default is `False`
- [ ] Model change is exactly 2 lines
- [ ] Hook contextmanager properly cleans up