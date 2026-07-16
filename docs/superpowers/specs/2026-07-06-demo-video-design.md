# Demo Video Export — Design Spec

**Date:** 2026-07-06  
**Status:** draft  
**Scope:** New standalone script `scripts/export_demo_video.py`

---

## 1. Purpose

Generate a side-by-side comparison video (MP4 + GIF) showing ground-truth vs. model-predicted human poses across a full 297-frame action sequence. Designed for presentation/reporting use.

## 2. Requirements

| # | Requirement | Detail |
|---|-------------|--------|
| R1 | **Layout** | Side-by-side: GT skeleton (left) + Pred skeleton (right) |
| R2 | **Rendering style** | Skeleton bones + joint scatter points (reuse `evaluation/pose_viz.py` primitives) |
| R3 | **Output formats** | MP4 (H.264) + animated GIF — both generated in one run |
| R4 | **Frame rate** | 18 FPS → ~16.5 s for 297 frames |
| R5 | **Resolution** | 1280×720 (720p), each sub-panel ~640×720 |
| R6 | **Overlay info** | Action name + Subject ID (top-left), real-time MPJPE in mm (top-right), frame counter + progress bar (bottom) |
| R7 | **Action selection** | Parameterized by action/sequence index — one video per run |
| R8 | **Model input** | CSI frames fed sequentially via DataLoader, no batching needed (B=1 per frame is fine) |
| R9 | **Zero codebase intrusion** | New standalone script; no changes to `eval.py`, `pose_viz.py`, or any existing module |

## 3. Technical Approach

**Chosen: Pure Python + matplotlib + ffmpeg** (Approach 1 from brainstorming).

Rationale:
- `evaluation/pose_viz.py` already provides `_draw_skeleton()` and `_draw_scatter()` — 90% reuse
- matplotlib is already a project dependency
- ffmpeg is a one-time `conda install ffmpeg` on the Linux server
- Full control over layout, annotations, and styling

## 4. Architecture & Data Flow

```
CLI args (action, checkpoint, dataset-root, ...)
         │
         ▼
┌─────────────────────────────────────┐
│ 1. load_checkpoint_model()          │  ← reuse from eval.py
│    Reconstruct WiFlowModel from     │
│    train_config + state_dict        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. build_dataset()                  │  ← MemmapDataset filtered by action
│    MemmapDataset → single-action    │
│    DataLoader (batch_size=1)        │
└──────────────┬──────────────────────┘
               │ 297 frames × CSI
               ▼
┌─────────────────────────────────────┐
│ 3. Inference loop                   │
│    for each frame:                  │
│      model(csi) → keypoints [18,2]  │
│      store (gt, pred, mpjpe)        │
└──────────────┬──────────────────────┘
               │ list of (gt, pred, mpjpe)
               ▼
┌─────────────────────────────────────┐
│ 4. Render frame PNGs                │
│    for each frame:                  │
│      _draw_skeleton(ax_left, gt)    │  ← reuse pose_viz primitives
│      _draw_skeleton(ax_right, pred) │
│      _draw_scatter(ax_left, gt)     │
│      _draw_scatter(ax_right, pred)  │
│      add overlay text + progress    │
│      fig.savefig(frame_N.png)       │
│      plt.close(fig)                 │
└──────────────┬──────────────────────┘
               │ 297 PNGs in temp dir
               ▼
┌─────────────────────────────────────┐
│ 5. Encode video                     │
│    ffmpeg -r 18 -i frame_%03d.png   │
│      → output.mp4 (H.264)           │
│      → output.gif (palettegen +     │
│        paletteuse)                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 6. Cleanup                          │
│    Remove temp PNG directory        │
│    (or keep with --keep-frames)     │
└─────────────────────────────────────┘
```

## 5. Script Design — `scripts/export_demo_video.py`

### 5.1 CLI Interface

```
python scripts/export_demo_video.py \
  --checkpoint    outputs/train/best_val_pck_0_2.pth \
  --dataset-root  data/mmfi_pose \
  --action        0 \
  --subject       1 \
  --output-dir    outputs/demo_videos/ \
  --fps           18 \
  --resolution    1280x720 \
  --model-label   "WiFlow v2 — Source Only" \
  [--keep-frames] [--gif-only] [--mp4-only]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | Path | **required** | `.pth` checkpoint file |
| `--dataset-root` | Path | **required** | Root of memmap dataset |
| `--action` | int | **required** | Action class index to visualize |
| `--subject` | int | `0` | Subject within the action (0 = first available) |
| `--output-dir` | Path | `outputs/demo_videos` | Output directory |
| `--fps` | float | `18` | Frames per second |
| `--resolution` | str | `1280x720` | W×H or named preset (`720p`, `1080p`) |
| `--model-label` | str | `"WiFlow v2"` | Label shown on prediction panel |
| `--keep-frames` | flag | `False` | Keep intermediate PNG frames |
| `--gif-only` | flag | `False` | Skip MP4, only GIF |
| `--mp4-only` | flag | `False` | Skip GIF, only MP4 |
| `--manifest-key` | str | `"test"` | Dataset split (`train`/`val`/`test`) |

### 5.2 Core Functions

```
def load_checkpoint_model(checkpoint_path, device)
    → WiFlowModel, train_config dict

def collect_action_frames(dataset, action, subject, device)
    → list[dict]  # [{csi, gt_keypoints, frame_idx}, ...]

def run_inference(model, frames, device)
    → list[dict]  # [{gt, pred, mpjpe}, ...]

def render_frame(gt, pred, mpjpe, frame_idx, total, action_label, subject_id, model_label, figure_size_dpi)
    → Path  # saved PNG

def encode_video(frame_dir, output_path, fps, output_type)
    → Path  # mp4 or gif

def main() → CLI entry point
```

### 5.3 Figure Layout (per frame)

```
┌─────────────────────────────────────────────────────┐
│  Action: Waving  │  Subject: S01    MPJPE: 12.4 mm  │  ← suptitle / annotation row
├──────────────────────┬──┬───────────────────────────┤
│                      │  │                           │
│    GT Skeleton       │  │    Pred Skeleton          │
│    (blue dashed      │  │    (red solid             │
│     + hollow ○)      │  │     + filled ●)           │
│                      │  │                           │
│    "Ground Truth"    │  │    "WiFlow v2"            │
│                      │  │                           │
├──────────────────────┴──┴───────────────────────────┤
│  ████████████████░░░░░░░░░░░░░░  Frame 148 / 297    │  ← progress bar
└─────────────────────────────────────────────────────┘
```

- **GT panel title:** "Ground Truth" (blue, top center of left subplot)
- **Pred panel title:** model_label value (red, top center of right subplot)
- **GT bones:** dashed `--`, `#4a9eff` blue, hollow circles
- **Pred bones:** solid `-`, `#ff6b6b` red, filled circles
- **Progress bar:** matplotlib `ax_bar` spanning bottom, filled portion = frame_idx / total

### 5.4 Reuse from `evaluation/pose_viz.py`

Direct imports (no modification needed):
- `BONE_EDGES` — 19 skeleton edges
- `JOINT_COLORS` — 18 joint colors
- `_draw_skeleton(ax, keypoints, hollow, ...)` — bone + joint rendering
- `_draw_scatter(ax, target, prediction)` — joint scatter with error vectors

_Note:_ `_draw_skeleton` and `_draw_scatter` are private (`_` prefixed). This is acceptable for a script in the same project. If we want to be cleaner, we could extract them to a shared `evaluation/draw_utils.py` — but that's scope creep; not doing it in this spec.

## 6. MPJPE Calculation

Per-frame MPJPE (Mean Per Joint Position Error):

```
mpjpe = ||gt - pred||₂  averaged over 18 joints (in mm, after denormalization)
```

This is the same metric used in `eval.py`. The demo script will compute it per-frame for the overlay display, not accumulate across frames.

## 7. Output Artifacts

```
outputs/demo_videos/
├── action_0_waving_subject_1_1280x720_18fps.mp4    (~2-8 MB)
├── action_0_waving_subject_1_1280x720_18fps.gif    (~10-40 MB)
└── frames_action_0/                                 (if --keep-frames)
    ├── frame_001.png
    ├── frame_002.png
    └── ...
```

File naming: `{action_label}_{subject_id}_{W}x{H}_{fps}fps.{ext}`

## 8. Dependencies

| Dependency | Available? | Action |
|------------|-----------|--------|
| matplotlib | ✅ (used by eval.py/pose_viz.py) | None |
| numpy | ✅ | None |
| torch | ✅ | None |
| ffmpeg | ❓ (system-level) | `conda install -c conda-forge ffmpeg` |
| Pillow | ✅ (matplotlib dep) | None |

## 9. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| ffmpeg not installed on server | Medium | Script checks `shutil.which("ffmpeg")` at startup, gives clear install instruction |
| Memory pressure from 297 × 1280×720 PNGs (~50 MB) | Low | Use `tempfile.mkdtemp` for frames dir; auto-cleanup unless `--keep-frames` |
| Private API reliance (`_draw_skeleton`) | Low | Documented; existing `export_report_pose_visualizations.py` already does this |
| Frame index mismatch between CSI and GT | Low | Dataset manifest guarantees alignment; script validates array lengths match |
| Matplotlib backend issues on headless server | Low | Script sets `matplotlib.use("Agg")` at top |

## 10. Testing Strategy

- **Unit test:** `tests/test_export_demo_video.py`
  - `test_render_frame_creates_png` — synthetic tensors, verify file exists + dimensions
  - `test_mpjpe_calculation` — known inputs → expected output
  - `test_collect_action_frames` — mock dataset, verify correct frame count (297)
  - `test_ffmpeg_available` — skip encoding tests if ffmpeg missing

## 11. Open Items

- [ ] Confirm ffmpeg is available on the Linux server (or install it)
- [ ] Decide: should `--action` accept action name string (e.g. "waving") or only index? (start with index; names can be mapped later)
- [ ] Confirm the 297 frame count is consistent across all actions in the MMFi dataset
