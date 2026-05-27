"""Cross-environment feature difference visualization.

Compares intermediate features between source and target domains to
identify CSI features affected by environmental conditions.

Requires ``--cross-env-viz`` flag on ``eval.py``.

Figures
-------
A. Channel Activation Delta   — mirrored bar chart + delta heatmap
B. Correlation Delta (Dr)     — 6×3 grid of r(source) - r(target)
C. Attention Offset           — source/target/delta per attention type
D. Feature Distribution Shift — PCA + MMD
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
from torch.utils.data import DataLoader

from dataloader import create_memmap_data_loader
from evaluation.hooks import wiflow_hooks
from evaluation.feature_viz import (
    _ANATOMY_COLORS,
    _ANATOMY_GROUPS,
    _GLOBAL_SPACING,
    _JOINT_NAMES,
    _add_colorbar,
    _apply_spacing,
    _save_fig,
    FONT_FAMILY,
)
from models import WiFlowModel
from train import prepare_model_input

# ---------------------------------------------------------------------------
# Feature collection
# ---------------------------------------------------------------------------

HOOK_POINTS = [
    "axial_encoder",
    "axial_encoder.spatial_attention",
    "axial_encoder.temporal_attention",
]


def _collect_env_features(
    loader: DataLoader,
    model: WiFlowModel,
    device: torch.device,
    source_env: str,
    target_env: str,
) -> tuple[
    dict[str, dict[str, list[torch.Tensor]]],
    dict[str, dict[str, list[torch.Tensor]]],
    dict[str, dict[str, list[torch.Tensor]]],
    dict[str, dict[str, list[torch.Tensor]]],
]:
    """Collect axial encoder features and attention weights grouped by (env, action).

    Returns:
        env_feats:  {env: {action: [features_gap]}}
        env_kpts:   {env: {action: [keypoints]}}
        env_sp_att: {env: {action: [spatial_attn_weights]}}
        env_tp_att: {env: {action: [temporal_attn_weights]}}
    """
    env_feats: dict[str, dict[str, list[torch.Tensor]]] = {
        source_env: defaultdict(list),
        target_env: defaultdict(list),
    }
    env_kpts: dict[str, dict[str, list[torch.Tensor]]] = {
        source_env: defaultdict(list),
        target_env: defaultdict(list),
    }
    env_sp_att: dict[str, dict[str, list[torch.Tensor]]] = {
        source_env: defaultdict(list),
        target_env: defaultdict(list),
    }
    env_tp_att: dict[str, dict[str, list[torch.Tensor]]] = {
        source_env: defaultdict(list),
        target_env: defaultdict(list),
    }

    with torch.no_grad():
        for batch in loader:
            model_input, target = prepare_model_input(batch, device)
            with wiflow_hooks(model, HOOK_POINTS) as ctx:
                _ = model(model_input)

            features = ctx.get_tensor("axial_encoder")
            sp_weights = ctx.get_attention_weights("axial_encoder.spatial_attention")
            tp_weights = ctx.get_attention_weights("axial_encoder.temporal_attention")

            if features is None:
                continue

            features_gap = features.mean(dim=[2, 3]).cpu()  # [B, 256]

            for i in range(features_gap.shape[0]):
                env = str(batch["environment"][i])
                if env not in (source_env, target_env):
                    continue
                action = str(batch["action"][i])

                env_feats[env][action].append(features_gap[i:i + 1])
                env_kpts[env][action].append(target[i:i + 1].cpu())

                if sp_weights is not None:
                    env_sp_att[env][action].append(sp_weights[i:i + 1].cpu())
                if tp_weights is not None:
                    env_tp_att[env][action].append(tp_weights[i:i + 1].cpu())

    return env_feats, env_kpts, env_sp_att, env_tp_att


def _compute_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> float:
    """Maximum Mean Discrepancy with RBF kernel."""
    xx = _rbf_kernel(x, x, sigma)
    yy = _rbf_kernel(y, y, sigma)
    xy = _rbf_kernel(x, y, sigma)
    return float(xx.mean() + yy.mean() - 2 * xy.mean())


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    xx = (x ** 2).sum(dim=1, keepdim=True)
    yy = (y ** 2).sum(dim=1, keepdim=True)
    dist = xx + yy.T - 2 * x @ y.T
    return torch.exp(-dist.clamp_min(0) / (2 * sigma * sigma))


# ---------------------------------------------------------------------------
# Figure A: Channel Activation Delta
# ---------------------------------------------------------------------------


def _figA_channel_activation_delta(
    env_feats: dict[str, dict[str, list[torch.Tensor]]],
    source_env: str,
    target_env: str,
    output_dir: Path,
) -> None:
    """Global mirrored bar chart of per-channel mean activations + delta heatmap."""
    src_all = torch.cat(
        [torch.cat(lst) for lst in env_feats[source_env].values()]
    ).numpy()  # [N_src, 256]
    tgt_all = torch.cat(
        [torch.cat(lst) for lst in env_feats[target_env].values()]
    ).numpy()  # [N_tgt, 256]

    src_mean = src_all.mean(axis=0)  # [256]
    tgt_mean = tgt_all.mean(axis=0)  # [256]
    delta = src_mean - tgt_mean  # [256]

    # Sort channels by descending absolute delta
    sort_idx = np.argsort(-np.abs(delta))
    src_sorted = src_mean[sort_idx]
    tgt_sorted = tgt_mean[sort_idx]
    delta_sorted = delta[sort_idx]

    fig, (ax_bar, ax_heat) = plt.subplots(
        2, 1, figsize=(14, 6),
        gridspec_kw={"height_ratios": [2.5, 1]},
    )

    # --- mirrored bar chart ---
    ch = np.arange(256)
    ax_bar.barh(ch, src_sorted, height=0.8, color=_ANATOMY_COLORS["upper"],
                alpha=0.75, label=f"{source_env} (mean)")
    ax_bar.barh(ch, -tgt_sorted, height=0.8, color=_ANATOMY_COLORS["lower"],
                alpha=0.75, label=f"{target_env} (mean)")
    ax_bar.axvline(0, color="#AAAAAA", linewidth=0.6)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Mean activation", color="#666666")
    ax_bar.legend(loc="lower right", fontsize=9, framealpha=0.85,
                  edgecolor="#DDDDDD", borderpad=0.8)
    ax_bar.set_title(
        f"Per-Channel Activation: {source_env} vs {target_env}",
        fontsize=12, fontweight="bold", color="#333333", pad=10,
    )
    ax_bar.tick_params(colors="#888888")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # --- delta heatmap ---
    delta_2d = delta_sorted.reshape(1, -1)
    vmax = max(abs(delta_2d.min()), abs(delta_2d.max()), 1e-6)
    im = ax_heat.imshow(
        delta_2d, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
    )
    ax_heat.set_yticks([])
    ax_heat.set_xlabel("Channel index (sorted by |Δ|)", color="#666666")
    ax_heat.tick_params(colors="#888888")
    _add_colorbar(im, ax_heat, label=f"Δ ({source_env} − {target_env})")

    _apply_spacing(fig)
    _save_fig(fig, output_dir / "_cross_env" / "figA_channel_delta")


def _figA_per_action(
    env_feats: dict[str, dict[str, list[torch.Tensor]]],
    source_env: str,
    target_env: str,
    output_dir: Path,
) -> None:
    """Per-action sparkline grid of channel-wise delta."""
    actions = sorted(
        set(env_feats[source_env].keys()) & set(env_feats[target_env].keys())
    )
    if len(actions) < 2:
        return

    n_actions = len(actions)
    fig, axes = plt.subplots(n_actions, 1, figsize=(14, max(2 * n_actions, 6)))
    if n_actions == 1:
        axes = [axes]

    global_vmax = 0.0
    action_deltas: dict[str, np.ndarray] = {}
    for action in actions:
        src = torch.cat(env_feats[source_env][action]).numpy()
        tgt = torch.cat(env_feats[target_env][action]).numpy()
        delta = src.mean(axis=0) - tgt.mean(axis=0)
        action_deltas[action] = delta
        global_vmax = max(global_vmax, float(np.abs(delta).max()))

    for ax, action in zip(axes, actions):
        delta = action_deltas[action]
        colors = [
            _ANATOMY_COLORS["upper"] if v >= 0 else _ANATOMY_COLORS["head"]
            for v in delta
        ]
        ax.bar(np.arange(256), delta, width=1.0, color=colors, alpha=0.7)
        ax.axhline(0, color="#AAAAAA", linewidth=0.4)
        ax.set_ylim(-global_vmax * 1.1, global_vmax * 1.1)
        ax.set_ylabel(action, fontsize=9, color="#444444", rotation=0,
                      ha="right", va="center", labelpad=50)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].set_xticks([0, 64, 128, 192, 255])
    axes[-1].set_xlabel("Channel index", color="#666666")
    axes[-1].tick_params(colors="#888888")

    fig.suptitle(
        f"Per-Action Channel Δ ({source_env} − {target_env})",
        fontsize=13, fontweight="bold", color="#333333",
    )
    _apply_spacing(fig)
    _save_fig(fig, output_dir / "_cross_env" / "figA_channel_delta_per_action")


# ---------------------------------------------------------------------------
# Figure B: Correlation Delta
# ---------------------------------------------------------------------------


def _figB_correlation_delta(
    env_feats: dict[str, dict[str, list[torch.Tensor]]],
    env_kpts: dict[str, dict[str, list[torch.Tensor]]],
    source_env: str,
    target_env: str,
    output_dir: Path,
) -> None:
    """6×3 grid of Δr = r(source) − r(target) per joint and spatial position."""
    src_all_feats = torch.cat(
        [torch.cat(lst) for lst in env_feats[source_env].values()]
    )
    src_all_kpts = torch.cat(
        [torch.cat(lst) for lst in env_kpts[source_env].values()]
    )
    tgt_all_feats = torch.cat(
        [torch.cat(lst) for lst in env_feats[target_env].values()]
    )
    tgt_all_kpts = torch.cat(
        [torch.cat(lst) for lst in env_kpts[target_env].values()]
    )

    # Compute r maps: [18, 256] per env (need full spatial features, not GAP)
    # For this figure we need spatial features, so we must re-run collection
    # We compute correlation with GAP features here; full spatial needs a
    # separate pass.  Use a simpler approach: correlate each of 256 channels
    # with each joint coordinate.

    src_feats_np = src_all_feats.numpy().astype(np.float64)  # [N_src, 256]
    src_kpts_np = src_all_kpts.numpy().astype(np.float64)    # [N_src, 18, 2]
    tgt_feats_np = tgt_all_feats.numpy().astype(np.float64)  # [N_tgt, 256]
    tgt_kpts_np = tgt_all_kpts.numpy().astype(np.float64)    # [N_tgt, 18, 2]

    fig, axes = plt.subplots(6, 3, figsize=(18, 12))
    axes = axes.flatten()

    for j in range(18):
        ax = axes[j]
        delta_r = np.zeros(256)

        for ch in range(256):
            if np.std(src_feats_np[:, ch]) < 1e-8 or np.std(tgt_feats_np[:, ch]) < 1e-8:
                continue
            r_src, _ = pearsonr(
                src_feats_np[:, ch],
                (src_kpts_np[:, j, 0] + src_kpts_np[:, j, 1]) * 0.5,
            )
            r_tgt, _ = pearsonr(
                tgt_feats_np[:, ch],
                (tgt_kpts_np[:, j, 0] + tgt_kpts_np[:, j, 1]) * 0.5,
            )
            delta_r[ch] = r_src - r_tgt

        delta_r_2d = delta_r.reshape(16, 16)
        vmax = max(abs(delta_r_2d.min()), abs(delta_r_2d.max()), 1e-6)
        im = ax.imshow(delta_r_2d, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto", origin="lower")
        ax.set_title(_JOINT_NAMES[j], fontsize=9, color="#333333")
        ax.set_xticks([])
        ax.set_yticks([])

        max_idx = np.unravel_index(np.argmax(np.abs(delta_r_2d)), delta_r_2d.shape)
        ax.annotate("×", (max_idx[1], max_idx[0]),
                    color="white", fontsize=8, ha="center", va="center",
                    fontweight="bold")

    # shared colorbar
    cbar_ax = fig.add_axes([0.94, 0.08, 0.010, 0.84])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f"Δr ({source_env} − {target_env})", fontsize=8, color="#555555")
    cbar.ax.tick_params(labelsize=7, colors="#888888")
    cbar.outline.set_visible(False)

    # anatomical group borders
    group_data = [
        ("head", slice(0, 6), _ANATOMY_COLORS["head"]),
        ("upper", slice(6, 12), _ANATOMY_COLORS["upper"]),
        ("lower", slice(12, 18), _ANATOMY_COLORS["lower"]),
    ]
    fig.canvas.draw()
    for _, sl, color in group_data:
        bbox0 = axes[sl.start].get_position()
        bbox1 = axes[sl.stop - 1].get_position()
        rect = plt.Rectangle(
            (bbox0.x0, bbox1.y0),
            bbox1.x1 - bbox0.x0,
            bbox0.y1 - bbox1.y0,
            fill=False, edgecolor=color, linewidth=1.2, linestyle="--",
            transform=fig.transFigure, clip_on=False,
        )
        fig.patches.append(rect)

    fig.suptitle(
        f"Channel-Pose Correlation Delta: {source_env} − {target_env}",
        fontsize=13, fontweight="bold", color="#333333",
    )
    _apply_spacing(fig)
    fig.subplots_adjust(right=0.93)
    _save_fig(fig, output_dir / "_cross_env" / "figB_correlation_delta")


# ---------------------------------------------------------------------------
# Figure C: Attention Offset
# ---------------------------------------------------------------------------


def _figC_attention_offset(
    env_sp_att: dict[str, dict[str, list[torch.Tensor]]],
    env_tp_att: dict[str, dict[str, list[torch.Tensor]]],
    source_env: str,
    target_env: str,
    output_dir: Path,
) -> None:
    """3×2 grid: source / target / delta for spatial and temporal attention."""
    def _avg_weights(att_dict, env):
        all_w = []
        for lst in att_dict[env].values():
            for w in lst:
                if w.ndim == 4:
                    all_w.append(w[0].mean(dim=0).numpy())  # avg heads
                elif w.ndim == 3:
                    all_w.append(w[0].numpy())
        return np.stack(all_w).mean(axis=0) if all_w else None

    src_sp = _avg_weights(env_sp_att, source_env)
    tgt_sp = _avg_weights(env_sp_att, target_env)
    src_tp = _avg_weights(env_tp_att, source_env)
    tgt_tp = _avg_weights(env_tp_att, target_env)

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))

    row_labels = [
        f"{source_env}",
        f"{target_env}",
        f"Δ ({source_env} − {target_env})",
    ]

    for row_idx, (sp, tp) in enumerate([
        (src_sp, src_tp),
        (tgt_sp, tgt_tp),
        (None, None),
    ]):
        if row_idx < 2:
            sp_max = max(abs(sp.min()), abs(sp.max()), 1e-6)
            tp_max = max(abs(tp.min()), abs(tp.max()), 1e-6)
            im0 = axes[row_idx, 0].imshow(sp, cmap="viridis", aspect="auto",
                                           vmin=0, vmax=sp_max)
            axes[row_idx, 0].set_ylabel(row_labels[row_idx], fontsize=10,
                                        color="#444444")
            im1 = axes[row_idx, 1].imshow(tp, cmap="viridis", aspect="auto",
                                           vmin=0, vmax=tp_max)
            _add_colorbar(im0, axes[row_idx, 0], label="attn weight")
            _add_colorbar(im1, axes[row_idx, 1], label="attn weight")
        else:
            if src_sp is not None and tgt_sp is not None:
                delta_sp = src_sp - tgt_sp
                dmax = max(abs(delta_sp.min()), abs(delta_sp.max()), 1e-6)
                im0 = axes[row_idx, 0].imshow(delta_sp, cmap="RdBu_r", aspect="auto",
                                              vmin=-dmax, vmax=dmax)
                _add_colorbar(im0, axes[row_idx, 0], label="Δ weight")
            else:
                axes[row_idx, 0].axis("off")

            if src_tp is not None and tgt_tp is not None:
                delta_tp = src_tp - tgt_tp
                dmax = max(abs(delta_tp.min()), abs(delta_tp.max()), 1e-6)
                im1 = axes[row_idx, 1].imshow(delta_tp, cmap="RdBu_r", aspect="auto",
                                              vmin=-dmax, vmax=dmax)
                _add_colorbar(im1, axes[row_idx, 1], label="Δ weight")
            else:
                axes[row_idx, 1].axis("off")

        axes[row_idx, 0].set_ylabel(row_labels[row_idx], fontsize=10, color="#444444")
        axes[row_idx, 0].tick_params(colors="#888888")
        axes[row_idx, 1].tick_params(colors="#888888")

    axes[0, 0].set_title("Spatial Attention", fontsize=12, fontweight="bold",
                          color="#333333")
    axes[0, 1].set_title("Temporal Attention", fontsize=12, fontweight="bold",
                          color="#333333")

    fig.suptitle(
        f"Attention Offset: {source_env} vs {target_env}",
        fontsize=13, fontweight="bold", color="#333333",
    )
    _apply_spacing(fig)
    _save_fig(fig, output_dir / "_cross_env" / "figC_attention_offset")


# ---------------------------------------------------------------------------
# Figure D: Feature Distribution Shift
# ---------------------------------------------------------------------------


def _figD_feature_distribution_shift(
    env_feats: dict[str, dict[str, list[torch.Tensor]]],
    source_env: str,
    target_env: str,
    output_dir: Path,
) -> None:
    """PCA 2D of GAP features colored by environment, with MMD annotation."""
    src_all = torch.cat(
        [torch.cat(lst) for lst in env_feats[source_env].values()]
    ).numpy()
    tgt_all = torch.cat(
        [torch.cat(lst) for lst in env_feats[target_env].values()]
    ).numpy()

    all_data = np.concatenate([src_all, tgt_all], axis=0)
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_data)
    n_src = src_all.shape[0]
    src_2d = all_2d[:n_src]
    tgt_2d = all_2d[n_src:]

    # MMD
    mmd_val = _compute_mmd(
        torch.from_numpy(src_all), torch.from_numpy(tgt_all)
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(src_2d[:, 0], src_2d[:, 1], c=_ANATOMY_COLORS["upper"],
               alpha=0.35, s=12, label=f"{source_env} ({n_src})")
    ax.scatter(tgt_2d[:, 0], tgt_2d[:, 1], c=_ANATOMY_COLORS["lower"],
               alpha=0.35, s=12, label=f"{target_env} ({tgt_all.shape[0]})")

    # 68% confidence ellipses
    for data, color, label in [
        (src_2d, _ANATOMY_COLORS["upper"], source_env),
        (tgt_2d, _ANATOMY_COLORS["lower"], target_env),
    ]:
        from matplotlib.patches import Ellipse
        mean = data.mean(axis=0)
        cov = np.cov(data.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * np.sqrt(eigvals)  # ≈ 95% CI, 1-sigma would be sqrt(eigvals)
        # Use 1-sigma (~68% CI)
        width, height = np.sqrt(eigvals)
        ellipse = Ellipse(
            xy=mean, width=width * 2, height=height * 2,
            angle=angle, fill=False, edgecolor=color,
            linewidth=1.2, linestyle="--",
        )
        ax.add_patch(ellipse)

    # MMD annotation
    ax.text(
        0.95, 0.95, f"MMD = {mmd_val:.4f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FAFAFA",
                  edgecolor="#DDDDDD", alpha=0.85),
    )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", color="#666666")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", color="#666666")
    ax.legend(fontsize=9, framealpha=0.85, edgecolor="#DDDDDD", borderpad=0.8,
              markerscale=1.5)
    ax.set_title(
        f"Encoder Feature Distribution: {source_env} vs {target_env}",
        fontsize=12, fontweight="bold", color="#333333", pad=10,
    )
    ax.tick_params(colors="#888888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _apply_spacing(fig)
    _save_fig(fig, output_dir / "_cross_env" / "figD_distribution_shift")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_cross_env_visualization(
    model: WiFlowModel,
    loader: DataLoader,
    dataset_root: str,
    output_dir: Path,
    device: torch.device,
    source_env: str = "env1",
    target_env: str = "env2",
    batch_size: int = 64,
    num_workers: int = 0,
) -> None:
    """Collect features across environments and generate all cross-env figures.

    Parameters
    ----------
    model : WiFlowModel in eval mode.
    loader : existing test DataLoader (consumed for feature collection).
    dataset_root : path to memmap dataset (for creating a fresh loader if needed).
    output_dir : base output directory; ``_cross_env/`` is created underneath.
    device : torch.device.
    source_env / target_env : environment names to compare.
    batch_size / num_workers : for fresh loader creation.
    """
    cross_dir = output_dir / "_cross_env"
    cross_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Collecting features for {source_env} vs {target_env}...")
    env_feats, env_kpts, env_sp_att, env_tp_att = _collect_env_features(
        loader, model, device, source_env, target_env
    )

    src_actions = list(env_feats[source_env].keys())
    tgt_actions = list(env_feats[target_env].keys())
    src_n = sum(len(lst) for lst in env_feats[source_env].values())
    tgt_n = sum(len(lst) for lst in env_feats[target_env].values())
    print(
        f"  Collected: {source_env}={src_n} samples ({len(src_actions)} actions), "
        f"{target_env}={tgt_n} samples ({len(tgt_actions)} actions)"
    )

    if src_n == 0 or tgt_n == 0:
        print(
            "  [SKIP] One environment has no samples — "
            f"{source_env}={src_n}, {target_env}={tgt_n}"
        )
        return

    # Figure A
    print("  Generating figA: Channel Activation Delta...")
    _figA_channel_activation_delta(env_feats, source_env, target_env, output_dir)
    common_actions = set(env_feats[source_env]) & set(env_feats[target_env])
    if len(common_actions) >= 2:
        _figA_per_action(env_feats, source_env, target_env, output_dir)

    # Figure B
    print("  Generating figB: Correlation Delta...")
    _figB_correlation_delta(env_feats, env_kpts, source_env, target_env, output_dir)

    # Figure C
    has_sp = any(
        any(w.ndim >= 3 for w in lst) for env in (source_env, target_env)
        for lst in env_sp_att[env].values()
    )
    if has_sp:
        print("  Generating figC: Attention Offset...")
        _figC_attention_offset(env_sp_att, env_tp_att, source_env, target_env, output_dir)
    else:
        print("  [SKIP] figC: no attention weights captured (model may not return weights)")

    # Figure D
    print("  Generating figD: Feature Distribution Shift...")
    _figD_feature_distribution_shift(env_feats, source_env, target_env, output_dir)

    print("  Cross-environment visualization complete.")
