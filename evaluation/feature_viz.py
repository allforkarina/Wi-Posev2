"""Research-grade feature visualizations for WiFlow model evaluation.

Generates 5 figures answering specific scientific questions about the model's
internal representations.  Requires ``--feature-viz`` flag on ``eval.py``.

Figures
-------
1. Antenna Channel Response Analysis   — CSI input vs antenna_mixer output
2. Symmetric Downsampling Trajectory    — PCA of resblock outputs
3. Axial Attention Maps                 — spatial + temporal attention weights
4. Joint Query Trajectory               — t-SNE + cosine similarity across layers
5. Feature-Pose Correlation Landscape   — Pearson r between encoder & joints
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from dataloader import create_memmap_data_loader
from evaluation.hooks import wiflow_hooks
from models import NUM_OPENPOSE_KEYPOINTS, OPENPOSE_BONE_EDGES, WiFlowModel
from train import extract_prediction_keypoints, prepare_model_input

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

FONT_FAMILY = "DejaVu Sans"
_ANATOMY_COLORS = {
    "head":   "#E05C30",
    "upper":  "#534AB7",
    "trunk":  "#1D9E75",
    "lower":  "#378ADD",
}
_ANATOMY_GROUPS = {
    "head":   [0, 1, 14, 15, 16, 17],
    "upper":  [2, 3, 4, 5, 6, 7],
    "trunk":  [8, 11],
    "lower":  [9, 10, 12, 13],
}
_JOINT_NAMES = [
    "Nose", "Neck", "RSh", "RElb", "RWr",
    "LSh", "LElb", "LWr", "RHip", "RKnee",
    "RAnk", "LHip", "LKnee", "LAnk", "REye",
    "LEye", "REar", "LEar",
]

_GLOBAL_SPACING = dict(
    hspace=0.45, wspace=0.35,
    left=0.07, right=0.93, top=0.92, bottom=0.06,
)

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "image.interpolation": "nearest",
})


def _apply_spacing(fig: plt.Figure) -> None:
    fig.subplots_adjust(**_GLOBAL_SPACING)


_OUTPUT_FORMAT = "both"
_FIGURE_WIDTH: float | None = None
_FIGURE_HEIGHT: float | None = None


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _OUTPUT_FORMAT in ("pdf", "both"):
        fig.savefig(str(path.with_suffix(".pdf")), dpi=300)
    if _OUTPUT_FORMAT in ("png", "both"):
        fig.savefig(str(path.with_suffix(".png")), dpi=300)
    plt.close(fig)


def _add_colorbar(im, ax: plt.Axes, label: str = "") -> None:
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    if label:
        cbar.set_label(label, fontsize=8)


def _group_for_joint(j: int) -> str:
    for name, indices in _ANATOMY_GROUPS.items():
        if j in indices:
            return name
    return "head"


# ---------------------------------------------------------------------------
# Sampling: action × environment stratified
# ---------------------------------------------------------------------------


def _collect_action_env_samples(
    loader: DataLoader,
    model: WiFlowModel,
    device: torch.device,
    num_per_action: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Collect *num_per_action* samples per action from distinct environments.

    Returns ``{action: {env: sample_dict}}`` where each sample has keys
    ``model_input``, ``target``, ``prediction``, ``action``, ``environment``,
    ``frame_idx``.
    """
    action_env: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    with torch.no_grad():
        for batch in loader:
            model_input, target = prepare_model_input(batch, device)
            preds = extract_prediction_keypoints(model(model_input)).cpu().numpy()

            for i in range(len(preds)):
                action = str(batch["action"][i])
                env = str(batch["environment"][i])

                env_dict = action_env[action]
                if env in env_dict:
                    continue
                if len(env_dict) >= num_per_action:
                    continue

                env_dict[env] = {
                    "model_input": model_input[i:i + 1],
                    "target": target[i:i + 1],
                    "prediction": preds[i],
                    "action": action,
                    "environment": env,
                    "frame_idx": int(batch["frame_idx"][i]),
                }

            # Do not break early — iterate the full test set to collect
            # samples from all action types.  Skips duplicate envs per action
            # so the loop naturally terminates after all actions are covered.

    return dict(action_env)


def _flatten_samples(
    action_env: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Flatten per-action dict into a list with unique sample IDs."""
    result: list[dict[str, Any]] = []
    for action, env_dict in sorted(action_env.items()):
        for idx, (env, sample) in enumerate(sorted(env_dict.items())):
            sid = f"{action}_{env}_s{idx}"
            sample["sample_id"] = sid
            result.append(sample)
    return result


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Figure 1: Antenna Channel Response Analysis
# ---------------------------------------------------------------------------


def _fig1_antenna_channel(
    sample: dict[str, Any],
    hook_ctx: Any,
    output_dir: Path,
) -> None:
    """1×7 grid: 3 raw CSI inputs | separator | 3 mixer outputs.

    Cross-correlation overlay on output panels to highlight antenna interaction.
    """
    csi = sample["model_input"][0].cpu().numpy()  # [3, 114, 64]
    mixer_val = hook_ctx.get("spatial_encoder.antenna_mixer")
    if mixer_val is None or not isinstance(mixer_val, torch.Tensor):
        print(f"    [WARN] antenna_mixer output not captured for {sample['sample_id']}")
        return
    mixer = mixer_val[0].cpu().numpy()  # [3, 64, 114] in conv layout
    mixer = mixer.transpose(0, 2, 1)     # → [3, 114, 64]

    # Weight-clip axis labels removed per spec (no "weight-clip" info)

    fig, axes = plt.subplots(1, 7, figsize=(18, 4))

    # --- symmetric vmin/vmax per group ---
    inp_abs = np.abs(csi)
    out_abs = np.abs(mixer)
    inp_vmax = max(np.percentile(inp_abs, 99), 1e-6)
    out_vmax = max(np.percentile(out_abs, 99), 1e-6)

    # --- input columns ---
    for ch in range(3):
        ax = axes[ch]
        im = ax.imshow(csi[ch], aspect="auto", cmap="RdBu_r",
                       vmin=-inp_vmax, vmax=inp_vmax, origin="lower")
        ax.set_title(f"Ant {ch + 1} (raw)", fontsize=11)
        ax.set_xlabel("time step")
        if ch == 0:
            ax.set_ylabel("subcarrier")
    # shared input colorbar on col 3
    _add_colorbar(im, axes[2], label="amplitude")

    # --- separator column ---
    sep_ax = axes[3]
    sep_ax.axis("off")
    sep_ax.text(0.5, 0.5, "antenna\nmixer  \u2192",
                ha="center", va="center", fontsize=11, fontweight="bold")

    # --- output columns ---
    im_last = None
    for ch in range(3):
        ax = axes[4 + ch]
        im = ax.imshow(mixer[ch], aspect="auto", cmap="RdBu_r",
                       vmin=-out_vmax, vmax=out_vmax, origin="lower")
        ax.set_title(f"Ch {ch + 1} (mixed)", fontsize=11)
        ax.set_xlabel("time step")
        if ch == 0:
            ax.set_ylabel("subcarrier")
        im_last = im

        # cross-corr overlay between adjacent channels
        if ch < 2:
            with np.errstate(invalid="ignore"):
                corr = np.array([
                    np.corrcoef(mixer[ch, sc], mixer[ch + 1, sc])[0, 1]
                    for sc in range(114)
                ])
            corr = np.nan_to_num(corr, nan=0.0)
            ax_corr = ax.twinx()
            ax_corr.plot(corr, np.arange(114), color="#E05C30",
                         linewidth=1.0, alpha=0.7)
            ax_corr.set_ylim(0, 113)
            ax_corr.set_yticks([])
            if ch == 2:
                ax_corr.set_ylabel("cross-corr", fontsize=8, color="#E05C30")

    # shared output colorbar on last output column
    if im_last is not None:
        _add_colorbar(im_last, axes[6], label="amplitude")

    fig.suptitle(
        f"Antenna Channel Response Analysis — "
        f"{sample['action']} / {sample['environment']}",
        fontsize=14, fontweight="bold",
    )
    _apply_spacing(fig)
    _save_fig(fig, output_dir / "fig1_antenna_channel")


# ---------------------------------------------------------------------------
# Figure 2: Symmetric Downsampling Trajectory (PCA)
# ---------------------------------------------------------------------------


def _fig2_downsampling_trajectory(
    sample: dict[str, Any],
    hook_ctx: Any,
    output_dir: Path,
) -> None:
    """2×4 grid.  Row 1: PCA RGB of first 8 channels.  Row 2: channel variance."""
    csi = sample["model_input"][0].cpu().numpy()  # [3, 114, 64]
    csi_img = csi.mean(axis=0)  # [114, 64]

    stage_keys = [
        "spatial_encoder.resblock1",
        "spatial_encoder.resblock2",
        "spatial_encoder.resblock3",
    ]
    stage_labels = [
        "ResBlock 1 output\n[64, 32, 57]",
        "ResBlock 2 output\n[128, 16, 29]",
        "ResBlock 3 output\n[128, 16, 29]",
    ]
    stage_colors = ["#534AB7", "#1D9E75", "#D85A30"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 6))

    # --- column 0: original CSI ---
    axes[0, 0].imshow(csi_img, aspect="auto", cmap="jet", origin="lower")
    axes[0, 0].set_title("CSI Input (mean ant)\n[3, 114, 64]", fontsize=11)
    axes[0, 0].set_xlabel("subcarrier axis")
    axes[0, 0].set_ylabel("time axis")
    axes[1, 0].axis("off")

    # --- columns 1–3: resblock outputs ---
    for col, (key, label, color) in enumerate(
        zip(stage_keys, stage_labels, stage_colors), start=1
    ):
        tensor = hook_ctx.get(key)
        if tensor is None or not isinstance(tensor, torch.Tensor):
            continue
        feat = tensor[0].cpu().numpy()  # [C, H, W]
        top8 = feat[:8]  # [8, H, W]
        h, w = top8.shape[1], top8.shape[2]
        flat = top8.reshape(8, -1).T  # [H*W, 8]

        pca = PCA(n_components=3)
        rgb = pca.fit_transform(flat).reshape(h, w, 3)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        axes[0, col].imshow(rgb)
        axes[0, col].set_title(label, fontsize=11)
        axes[0, col].set_xlabel("subcarrier axis")
        axes[0, col].set_ylabel("time axis")

        # --- variance histogram ---
        chan_var = feat.var(axis=(1, 2))
        mean_var = float(chan_var.mean())
        axes[1, col].hist(chan_var, bins=32, color=color, alpha=0.75, edgecolor="white")
        axes[1, col].axvline(mean_var, color="black", linestyle="--", linewidth=1)
        axes[1, col].text(
            0.95, 0.95, f"mean={mean_var:.3f}",
            transform=axes[1, col].transAxes,
            ha="right", va="top", fontsize=8,
        )
        axes[1, col].set_xlabel("channel activation variance")
        if col == 1:
            axes[1, col].set_ylabel("count")

    fig.suptitle(
        f"Symmetric Downsampling Trajectory — "
        f"{sample['action']} / {sample['environment']}",
        fontsize=14, fontweight="bold",
    )
    _apply_spacing(fig)
    _save_fig(fig, output_dir / "fig2_downsampling_trajectory")


# ---------------------------------------------------------------------------
# Figure 3: Axial Attention Maps
# ---------------------------------------------------------------------------


def _fig3_axial_attention(
    sample: dict[str, Any],
    hook_ctx: Any,
    output_dir: Path,
) -> None:
    """2×5 grid: spatial (row 1) + temporal (row 2), avg + 4 heads each."""
    # MultiheadAttention with batch_first=True + need_weights=True
    # returns (output [B*T, N, D], weights [B*T, num_heads, N, N])
    sp_weights = hook_ctx.get_attention_weights("axial_encoder.spatial_attention")
    tp_weights = hook_ctx.get_attention_weights("axial_encoder.temporal_attention")

    if sp_weights is None and tp_weights is None:
        print(f"    [WARN] No attention weights captured for {sample['sample_id']}")
        return

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    row_colors = ["Blues", "Oranges"]

    for row_idx, (weights, label, n_tokens) in enumerate([
        (sp_weights, "Spatial", 29),
        (tp_weights, "Temporal", 16),
    ]):
        if weights is None:
            for c in range(5):
                axes[row_idx, c].axis("off")
                axes[row_idx, c].text(0.5, 0.5, "N/A", ha="center", va="center")
            continue

        # weights: [B*t, num_heads, L, L] (per‑head) or [B*t, L, L] (averaged)
        w = weights[0].cpu().numpy()  # [num_heads, L, L] or [L, L]
        if w.ndim not in (2, 3):
            print(f"    [WARN] Unexpected attention weight shape: {w.shape}")
            for c in range(5):
                axes[row_idx, c].axis("off")
            continue

        # Detect averaged vs per‑head
        if w.ndim == 2:
            # Already averaged across heads — show as single panel
            num_heads = 0
            avg_w = w  # [L, L]
        else:
            num_heads = w.shape[0]
            if num_heads < 1:
                print(f"    [WARN] num_heads={num_heads} for {label} attention")
                for c in range(5):
                    axes[row_idx, c].axis("off")
                continue
            avg_w = w.mean(axis=0)  # [L, L]

        # column 0: average
        _draw_attn_panel(
            axes[row_idx, 0], avg_w,
            title=f"{label} attn (avg {max(num_heads, 1)} heads)",
            token_label="subcarrier token" if label == "Spatial" else "time token",
            n=n_tokens,
        )

        # columns 1–4: individual heads (up to 4); skip if averaged
        if num_heads > 0:
            for h in range(min(4, num_heads)):
                _draw_attn_panel(
                    axes[row_idx, 1 + h], w[h],
                    title=f"{label} head {h + 1}",
                    token_label="subcarrier token" if label == "Spatial" else "time token",
                    n=n_tokens,
                )

        # hide unused head columns
        for h in range(num_heads if num_heads > 0 else 1, 4):
            axes[row_idx, 1 + h].axis("off")

    fig.suptitle(
        f"Axial Attention Maps — {sample['action']} / {sample['environment']}",
        fontsize=14, fontweight="bold",
    )
    _apply_spacing(fig)
    _save_fig(fig, output_dir / "fig3_axial_attention")


def _draw_attn_panel(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    token_label: str,
    n: int,
) -> None:
    """Draw one attention matrix + entropy bar on a panel."""
    if matrix.ndim != 2:
        print(f"    [WARN] _draw_attn_panel received ndim={matrix.ndim} shape={matrix.shape} — skipping")
        ax.axis("off")
        return

    im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(token_label)
    ax.set_ylabel(token_label)

    # tick every 5 tokens
    ticks = list(range(0, n, max(1, n // 5)))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # diagonal
    if matrix.shape[0] == matrix.shape[1]:
        ax.axline((0, 0), (1, 1), color="white", linestyle="--", linewidth=0.8)

    # colorbar
    _add_colorbar(im, ax, label="attn weight")


# ---------------------------------------------------------------------------
# Figure 4: Joint Query Trajectory (joint / hierarchical decoder)
# ---------------------------------------------------------------------------


def _fig4_joint_query_trajectory(
    sample: dict[str, Any],
    hook_ctx: Any,
    output_dir: Path,
) -> None:
    """2×(L+1) grid. Row 1: t-SNE of joint queries. Row 2: cosine similarity."""
    layer_queries: list[np.ndarray] = []
    layer_idx = 0
    while True:
        key = f"decoder.cross_attention_layers.{layer_idx}"
        tensor = hook_ctx.get(key)
        if tensor is None or not isinstance(tensor, torch.Tensor):
            break
        layer_queries.append(tensor[0].cpu().numpy())  # [18, 256]
        layer_idx += 1

    L = len(layer_queries)
    if L == 0:
        print(f"    [WARN] No decoder layer queries captured for {sample['sample_id']}")
        return

    ncols = L + 1  # +1 for legend/annotation column
    fig, axes = plt.subplots(2, ncols, figsize=(18, 10))

    # --- t-SNE ---
    all_q = np.concatenate(layer_queries, axis=0)  # [18*L, 256]
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    all_tsne = tsne.fit_transform(all_q)  # [18*L, 2]

    # global range with 10% margin
    global_min = all_tsne.min(axis=0) - 0.1 * (all_tsne.max(axis=0) - all_tsne.min(axis=0))
    global_max = all_tsne.max(axis=0) + 0.1 * (all_tsne.max(axis=0) - all_tsne.min(axis=0))

    for l in range(L):
        ax = axes[0, l]
        pts = all_tsne[l * 18:(l + 1) * 18]
        for group_name, indices in _ANATOMY_GROUPS.items():
            color = _ANATOMY_COLORS[group_name]
            ax.scatter(
                pts[indices, 0], pts[indices, 1],
                c=color, s=80, linewidths=0.5, edgecolors="white",
                label=group_name,
            )
        # joint index labels
        for j in range(18):
            ax.annotate(
                str(j), (pts[j, 0], pts[j, 1]),
                fontsize=8, alpha=0.7,
                xytext=(3, 3), textcoords="offset points",
            )
        ax.set_title(f"Layer {l} queries (t-SNE)", fontsize=10)
        ax.set_xlim(global_min[0], global_max[0])
        ax.set_ylim(global_min[1], global_max[1])
        ax.set_xticks([])
        ax.set_yticks([])

    # legend in last column
    leg_ax = axes[0, -1]
    leg_ax.axis("off")
    for group_name, color in _ANATOMY_COLORS.items():
        leg_ax.scatter([0.1], [0.8 - 0.15 * list(_ANATOMY_COLORS).index(group_name)],
                       c=color, s=80, linewidths=0.5, edgecolors="white")
        leg_ax.text(0.25, 0.8 - 0.15 * list(_ANATOMY_COLORS).index(group_name),
                    group_name, fontsize=10, va="center")
    leg_ax.text(0.1, 0.95, "Groups", fontsize=11, fontweight="bold")

    # --- cosine similarity ---
    for l in range(L):
        ax = axes[1, l]
        q = layer_queries[l].astype(np.float64)  # [18, 256]
        norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        sim = norm @ norm.T  # [18, 18]
        im = ax.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(f"Query similarity (layer {l})", fontsize=10)
        ax.set_xticks(range(0, 18, 3))
        ax.set_xticklabels([_JOINT_NAMES[j] for j in range(0, 18, 3)],
                           rotation=45, fontsize=7)
        ax.set_yticks(range(0, 18, 3))
        ax.set_yticklabels([_JOINT_NAMES[j] for j in range(0, 18, 3)], fontsize=7)
        _add_colorbar(im, ax, label="cosine similarity")

    # last column of row 2: hide
    axes[1, -1].axis("off")

    fig.suptitle(
        f"Joint Query Trajectory — {sample['action']} / {sample['environment']}",
        fontsize=14, fontweight="bold",
    )
    _apply_spacing(fig)
    _save_fig(fig, output_dir / "fig4_joint_query_trajectory")


# ---------------------------------------------------------------------------
# Figure 6: Feature-Pose Correlation Landscape (global)
# ---------------------------------------------------------------------------


def _fig6_global_correlation(
    model: WiFlowModel,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> None:
    """6×3 grid: Pearson r between encoder features and joint coordinates.

    Collects the whole test set — one pass.  Uses a separate DataLoader to
    avoid interfering with the per-sample visualization pass.
    """
    all_features: list[np.ndarray] = []
    all_keypoints: list[np.ndarray] = []

    hook_points = ["axial_encoder"]
    with torch.no_grad():
        for batch in loader:
            model_input, target = prepare_model_input(batch, device)
            with wiflow_hooks(model, hook_points) as ctx:
                _ = model(model_input)
            feats = ctx.get_tensor("axial_encoder")
            if feats is not None:
                all_features.append(feats.cpu().numpy())  # [B, 256, 29, 16]
            all_keypoints.append(target.cpu().numpy())  # [B, 18, 2]

    if not all_features:
        print("    [WARN] No encoder features collected for fig6")
        return

    features = np.concatenate(all_features, axis=0)  # [N, 256, 29, 16]
    keypoints = np.concatenate(all_keypoints, axis=0)  # [N, 18, 2]

    # mean pool over 256 channels → [N, 29, 16]
    feat_pooled = features.mean(axis=1)

    fig, axes = plt.subplots(6, 3, figsize=(18, 12))
    axes = axes.flatten()

    # anatomical group borders
    group_ranges = [
        ("head",   slice(0, 6)),
        ("upper",  slice(6, 12)),
        ("lower",  slice(12, 18)),
    ]

    for j in range(18):
        ax = axes[j]
        corr_x = np.zeros((29, 16))
        corr_y = np.zeros((29, 16))
        for r in range(29):
            for c in range(16):
                vals = feat_pooled[:, r, c]
                if np.std(vals) < 1e-8:
                    continue
                corr_x[r, c], _ = pearsonr(vals, keypoints[:, j, 0])
                corr_y[r, c], _ = pearsonr(vals, keypoints[:, j, 1])

        corr_avg = (corr_x + corr_y) / 2.0
        im = ax.imshow(corr_avg, cmap="coolwarm", vmin=-1, vmax=1,
                       aspect="auto", origin="lower")
        ax.set_title(_JOINT_NAMES[j], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        # white cross at max |r|
        max_idx = np.unravel_index(np.argmax(np.abs(corr_avg)), corr_avg.shape)
        ax.annotate("\u00d7", (max_idx[1], max_idx[0]),
                    color="white", fontsize=8, ha="center", va="center",
                    fontweight="bold")

        # axis labels only on outer edges
        if j >= 15:
            ax.set_xlabel("time token")
        if j % 3 == 0:
            ax.set_ylabel("subcarrier token")

    # shared colorbar
    cbar_ax = fig.add_axes([0.94, 0.08, 0.012, 0.84])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Pearson r", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    # anatomical group borders (dashed rectangles around rows)
    group_colors = [_ANATOMY_COLORS["head"], _ANATOMY_COLORS["upper"], _ANATOMY_COLORS["lower"]]
    for (group_name, sl), color in zip(group_ranges, group_colors):
        start_ax = axes[sl.start]
        end_ax = axes[sl.stop - 1]
        # get bounding box in figure coordinates
        fig.canvas.draw()
        bbox0 = start_ax.get_position()
        bbox1 = end_ax.get_position()
        rect = plt.Rectangle(
            (bbox0.x0, bbox1.y0),
            bbox1.x1 - bbox0.x0,
            bbox0.y1 - bbox1.y0,
            fill=False, edgecolor=color, linewidth=1.5, linestyle="--",
            transform=fig.transFigure, clip_on=False,
        )
        fig.patches.append(rect)

    fig.suptitle("Feature-Pose Correlation Landscape (Encoder Output \u00d7 Joint Coordinates)",
                 fontsize=14, fontweight="bold")
    _apply_spacing(fig)
    # Adjust right margin for shared colorbar
    fig.subplots_adjust(right=0.93)
    _save_fig(fig, output_dir / "_global" / "fig6_feature_pose_correlation")


# ---------------------------------------------------------------------------
# Shared thumbnail helper for composite figures
# ---------------------------------------------------------------------------

_THUMBNAIL_MAX_PX = 600


def _thumbnail(img_path: Path) -> Image.Image | None:
    """Load and resize to a uniform thumbnail; return None on failure."""
    from PIL import Image

    try:
        img = Image.open(img_path)
    except Exception:
        return None
    w, h = img.size
    if max(w, h) > _THUMBNAIL_MAX_PX:
        scale = _THUMBNAIL_MAX_PX / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _imshow_thumb(ax: plt.Axes, img_path: Path) -> None:
    """Load thumbnail and display it with aspect='auto' (fill subplot)."""
    img = _thumbnail(img_path)
    if img is not None:
        ax.imshow(img, aspect="auto")


# ---------------------------------------------------------------------------
# Overview composite
# ---------------------------------------------------------------------------


def _build_overview(
    sample_dirs: list[Path],
    output_dir: Path,
    decoder_type: str,
) -> None:
    """Build an overview thumbnail composite of all figures.

    Dynamically sizes the grid so each panel preserves its image's aspect
    ratio.  Uses shared thumbnail helper for consistent sizing.
    """
    try:
        from PIL import Image
    except ImportError:
        print("    [INFO] PIL not installed — skipping overview composite")
        return

    fig_names = [
        "fig1_antenna_channel",
        "fig2_downsampling_trajectory",
        "fig3_axial_attention",
        "fig4_joint_query_trajectory",
        "fig6_feature_pose_correlation",
    ]

    # Collect thumbnail paths
    thumb_paths: list[Path | None] = []
    for name in fig_names:
        if "fig6" in name:
            p = output_dir / "_global" / f"{name}.png"
            thumb_paths.append(p if p.exists() else None)
        else:
            found = False
            for d in sample_dirs:
                candidate = d / f"{name}.png"
                if candidate.exists():
                    thumb_paths.append(candidate)
                    found = True
                    break
            if not found:
                thumb_paths.append(None)

    valid = [p for p in thumb_paths if p is not None]
    if len(valid) < 3:
        return

    # Measure actual aspect ratios of thumbnails
    aspects: list[float] = []
    thumb_imgs: list[Image.Image] = []
    for p in thumb_paths:
        img = _thumbnail(p) if p is not None else None
        thumb_imgs.append(img)
        if img is not None:
            aspects.append(img.width / max(img.height, 1))
        else:
            aspects.append(1.0)

    # Layout: figure width is fixed; row heights adapt to content
    ncols = 3
    nrows = int(np.ceil(len(thumb_paths) / ncols))
    panel_w = _FIGURE_WIDTH / ncols if _FIGURE_WIDTH else 6.0

    # Build per-row heights based on the flattest subplot in each row
    row_heights: list[float] = []
    for r in range(nrows):
        row_aspects = aspects[r * ncols:(r + 1) * ncols]
        # flattest = smallest aspect (widest image needs most height)
        min_asp = min(row_aspects) if row_aspects else 1.0
        row_heights.append(panel_w / max(min_asp, 0.2))
    total_h = sum(row_heights) + 1.0  # +1" for suptitle margin

    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, total_h))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if idx >= len(thumb_paths) or thumb_imgs[idx] is None:
            ax.axis("off")
            continue
        ax.imshow(thumb_imgs[idx], aspect="auto")
        ax.set_title(f"Fig {idx + 1}", fontsize=12, fontweight="bold", loc="left")
        ax.axis("off")

    fig.suptitle("Feature Visualization Overview", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, output_dir / "overview")


# ---------------------------------------------------------------------------
# Per-action composite figures
# ---------------------------------------------------------------------------


def _build_action_composites(
    action_env: dict[str, dict[str, dict[str, Any]]],
    viz_dir: Path,
    model: WiFlowModel,
    all_hooks: list[str],
    device: torch.device,
) -> None:
    """For each action, generate a 1×N composite: axial attention.

    Each action gets one composite page showing fig3 from up to
    2 representative environment samples for side-by-side comparison.
    """
    try:
        from PIL import Image
    except ImportError:
        print("    [INFO] PIL not installed — skipping action composites")
        return

    actions_dir = viz_dir / "_actions"
    actions_dir.mkdir(parents=True, exist_ok=True)

    for action, env_dict in sorted(action_env.items()):
        print(f"  Building action composite: {action}")
        envs = sorted(env_dict.keys())
        n_envs = len(envs)

        # Take up to 2 environments for side-by-side comparison
        envs_to_show = envs[:2]

        # Ensure figures exist for these samples via per-sample loop results
        # Each sample should already have fig3 generated.  If not,
        # generate it now.
        sample_paths: list[Path] = []
        for env in envs_to_show:
            s_dir = viz_dir / f"{action}_{env}_s0"
            s_dir.mkdir(parents=True, exist_ok=True)
            sample_paths.append(s_dir)

            sample = env_dict[env]
            if not (s_dir / "fig3_axial_attention.png").exists():
                with wiflow_hooks(model, all_hooks) as ctx:
                    with torch.no_grad():
                        _ = model(sample["model_input"].to(device))
                if not (s_dir / "fig3_axial_attention.png").exists():
                    _fig3_axial_attention(sample, ctx, s_dir)

        # Build the composite: rows = figure types, cols = environment samples
        n_cols = len(envs_to_show)
        n_rows = 1  # fig3 only
        panel_w = _FIGURE_WIDTH / n_cols if _FIGURE_WIDTH else 6.0
        panel_h = panel_w  # square-ish panels, imshow with aspect='auto' handles fill

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(panel_w * n_cols, panel_h * n_rows + 0.8),
        )
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        fig_types = [
            ("fig3_axial_attention", "Axial Attention"),
        ]
        for r, (fname, f_label) in enumerate(fig_types):
            for c, sp in enumerate(sample_paths):
                ax = axes[r, c]
                img_path = sp / f"{fname}.png"
                if img_path.exists():
                    _imshow_thumb(ax, img_path)
                ax.set_title(
                    f"{envs_to_show[c]} — {f_label}",
                    fontsize=10, fontweight="bold",
                )
                ax.axis("off")

        fig.suptitle(
            f"Action: {action}  ({n_envs} environments sampled)",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        _save_fig(fig, actions_dir / f"composite_{action}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_feature_visualization(
    model: WiFlowModel,
    loader: DataLoader,
    dataset_root: str,
    output_dir: Path,
    device: torch.device,
    decoder_type: str,
    num_action_samples: int = 3,
    batch_size: int = 64,
    num_workers: int = 0,
    output_format: str = "both",
    figure_width: float | None = None,
    figure_height: float | None = None,
) -> None:
    """Orchestrate all 6 feature visualization figures.

    Parameters
    ----------
    model : WiFlowModel
        Trained model in eval mode.
    loader : DataLoader
        Test DataLoader (will be consumed once for sampling, then a fresh
        loader is created internally for the global correlation pass).
    dataset_root : str
        Path to the NPY memmap dataset directory (for creating fresh loader).
    output_dir : Path
        Base output directory; ``feature_viz/`` is created underneath.
    device : torch.device
    decoder_type : str
        One of ``"joint"``, ``"hierarchical"``.
    num_action_samples : int
        Samples per action type (default 3).
    batch_size : int
        Batch size for the fresh correlation loader.
    num_workers : int
        Number of data loader workers.
    output_format : str
        ``"png"``, ``"pdf"``, or ``"both"`` (default).
    figure_width : float | None
        Override default figure width in inches.
    figure_height : float | None
        Override default figure height in inches.
    """
    global _OUTPUT_FORMAT, _FIGURE_WIDTH, _FIGURE_HEIGHT
    _OUTPUT_FORMAT = output_format
    _FIGURE_WIDTH = figure_width
    _FIGURE_HEIGHT = figure_height

    viz_dir = output_dir / "feature_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # --- Sampling ---
    print("  Collecting stratified samples...")
    action_env = _collect_action_env_samples(loader, model, device, num_action_samples)
    n_samples = sum(len(e) for e in action_env.values())
    n_actions = len(action_env)
    print(f"  Sampled {n_samples} frames from {n_actions} actions")
    samples = _flatten_samples(action_env)

    # --- Hook points ---
    common_hook_points = [
        "spatial_encoder.antenna_mixer",
        "spatial_encoder.feature_stem",
        "spatial_encoder.resblock1",
        "spatial_encoder.resblock2",
        "spatial_encoder.resblock3",
        "axial_encoder.spatial_attention",
        "axial_encoder.temporal_attention",
        "axial_encoder",
    ]

    if decoder_type == "joint":
        decoder_hooks = [
            "decoder.cross_attention_layers.0",
            "decoder.cross_attention_layers.1",
            "decoder.cross_attention_layers.2",
        ]
    elif decoder_type == "hierarchical":
        decoder_hooks = [
            "decoder.stages.0",
            "decoder.stages.1",
            "decoder.stages.2",
        ]
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}")

    all_hooks = common_hook_points + decoder_hooks

    # --- Per-sample figures 1–5 ---
    sample_dirs: list[Path] = []
    for sample in samples:
        sample_dir = viz_dir / sample["sample_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_dirs.append(sample_dir)
        print(f"  Visualizing: {sample['sample_id']}")

        with wiflow_hooks(model, all_hooks) as ctx:
            with torch.no_grad():
                _ = model(sample["model_input"].to(device))

            # Fig 1: Antenna Channel Response
            _fig1_antenna_channel(sample, ctx, sample_dir)

            # Fig 2: Downsampling Trajectory
            _fig2_downsampling_trajectory(sample, ctx, sample_dir)

            # Fig 3: Axial Attention
            _fig3_axial_attention(sample, ctx, sample_dir)


            # Fig 4: Joint Query Trajectory (joint / hierarchical only)
            if decoder_type in ("joint", "hierarchical"):
                _fig4_joint_query_trajectory(sample, ctx, sample_dir)

    # --- Per-action composites ---
    print("  Building per-action composite figures...")
    _build_action_composites(
        action_env=action_env,
        viz_dir=viz_dir,
        model=model,
        all_hooks=all_hooks,
        device=device,
    )

    # --- Global figure 6 ---
    # Create a fresh loader for the one-pass correlation collection
    print("  Computing global feature-pose correlation (one pass over test set)...")
    global_loader = create_memmap_data_loader(
        data_dir=dataset_root,
        split="test",
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    global_dir = viz_dir / "_global"
    global_dir.mkdir(parents=True, exist_ok=True)
    _fig6_global_correlation(model, global_loader, device, viz_dir)

    # --- Overview ---
    print("  Building overview composite...")
    _build_overview(sample_dirs, viz_dir, decoder_type)