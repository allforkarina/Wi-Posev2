"""Compact attention evidence for the proposed axial encoder and joint decoder."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.memmap_dataset import MemmapDataset
from data.pose_schema import JOINT_NAMES
from dataloader import memmap_collate_fn
from models import WiFlowModel
from models.wiflow_joint_decoder import WiFlowJointDecoder
from train import prepare_model_input


def _choose_two_actions(dataset: MemmapDataset, seed: int) -> list[int]:
    grouped: dict[str, list[int]] = {}
    for position in range(len(dataset)):
        dataset_index = int(dataset.indices[position])
        action = str(dataset._actions[dataset_index])
        grouped.setdefault(action, []).append(position)
    if len(grouped) < 2:
        raise ValueError("Mechanism visualization requires at least two actions")
    generator = np.random.default_rng(seed)
    actions = sorted(grouped)
    selected_actions = generator.choice(actions, size=2, replace=False)
    return [
        grouped[str(action)][int(generator.integers(0, len(grouped[str(action)])))]
        for action in selected_actions
    ]


def _attention_evidence(
    model: WiFlowModel,
    model_input: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model.axial_mode != "spatial_then_temporal":
        raise ValueError("Mechanism export requires spatial_then_temporal axial mode")
    if not isinstance(model.decoder, WiFlowJointDecoder):
        raise ValueError("Mechanism export requires a joint-query decoder")

    calibrated = model.input_calibration(model_input)
    from models.csi_feature_bank import build_csi_feature_bank

    feature_bank = build_csi_feature_bank(calibrated, mode=model.csi_feature_mode)
    spatial_features = model.spatial_encoder(feature_bank)
    axial = model.axial_encoder
    batch_size, _, spatial_count, temporal_count = spatial_features.shape

    spatial_input = axial._prepare_spatial_attention_input(spatial_features)
    spatial_output, spatial_weights = axial.spatial_attention(
        spatial_input,
        spatial_input,
        spatial_input,
        need_weights=True,
        average_attn_weights=False,
    )
    spatial_output = axial.spatial_norm(spatial_output + spatial_input)
    spatial_after = axial._restore_spatial_attention_output(
        spatial_output,
        batch_size,
        spatial_count,
        temporal_count,
    )
    spatial_importance = (
        spatial_weights.mean(dim=(1, 2))
        .reshape(batch_size, temporal_count, spatial_count)
        .mean(dim=1)
    )

    temporal_input = axial._prepare_temporal_attention_input(spatial_after)
    temporal_output, temporal_weights = axial.temporal_attention(
        temporal_input,
        temporal_input,
        temporal_input,
        need_weights=True,
        average_attn_weights=False,
    )
    temporal_output = axial.temporal_norm(temporal_output + temporal_input)
    temporal_after = axial._restore_temporal_attention_output(
        temporal_output,
        batch_size,
        spatial_count,
        temporal_count,
    )
    temporal_importance = (
        temporal_weights.mean(dim=(1, 2))
        .reshape(batch_size, spatial_count, temporal_count)
        .mean(dim=1)
    )

    encoded = axial.channel_projection(temporal_after)
    decoder = model.decoder
    tokens = decoder.flatten_tokens(encoded)
    queries = decoder.joint_queries.unsqueeze(0).expand(batch_size, -1, -1)
    query_weights = None
    for layer in decoder.cross_attention_layers:
        attended, query_weights = layer.cross_attention(
            queries,
            tokens,
            tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        queries = layer.cross_attention_norm(queries + attended)
        queries = layer.feedforward_norm(queries + layer.feedforward(queries))
    if query_weights is None:
        raise RuntimeError("No decoder cross-attention weights were produced")
    query_token = query_weights.mean(dim=1).reshape(
        batch_size,
        len(JOINT_NAMES),
        spatial_count,
        temporal_count,
    ).mean(dim=2)
    return (
        spatial_importance.detach().cpu().numpy(),
        temporal_importance.detach().cpu().numpy(),
        query_token.detach().cpu().numpy(),
    )


def export_mechanism_visualization(
    model: WiFlowModel,
    dataset: MemmapDataset,
    device: torch.device,
    output_dir: Path,
    *,
    seed: int = 42,
    dpi: int = 150,
) -> Path:
    positions = _choose_two_actions(dataset, seed)
    loader = DataLoader(
        Subset(dataset, positions),
        batch_size=2,
        shuffle=False,
        collate_fn=memmap_collate_fn,
    )
    batch = next(iter(loader))
    model_input, _ = prepare_model_input(batch, device)
    with torch.no_grad():
        spatial, temporal, query_token = _attention_evidence(model, model_input)
    labels = [str(value) for value in batch["action"]]
    spatial = np.concatenate([spatial, np.abs(spatial[:1] - spatial[1:2])], axis=0)
    temporal = np.concatenate([temporal, np.abs(temporal[:1] - temporal[1:2])], axis=0)
    query_token = np.concatenate([
        query_token,
        np.abs(query_token[:1] - query_token[1:2]),
    ], axis=0)
    row_labels = [*labels, f"|{labels[0]} - {labels[1]}|"]

    figure, axes = plt.subplots(3, 3, figsize=(13, 10))
    for row, label in enumerate(row_labels):
        axes[row, 0].imshow(spatial[row][None, :], aspect="auto", cmap="viridis")
        axes[row, 0].set_ylabel(label)
        axes[row, 0].set_yticks([])
        axes[row, 0].set_xlabel("Spatial token")
        axes[row, 1].imshow(temporal[row][None, :], aspect="auto", cmap="magma")
        axes[row, 1].set_yticks([])
        axes[row, 1].set_xlabel("Temporal token")
        axes[row, 2].imshow(query_token[row], aspect="auto", cmap="cividis")
        axes[row, 2].set_yticks(range(len(JOINT_NAMES)))
        axes[row, 2].set_yticklabels(JOINT_NAMES, fontsize=6)
        axes[row, 2].set_xlabel("Temporal token")
    axes[0, 0].set_title("Spatial attention key importance")
    axes[0, 1].set_title("Temporal attention key importance")
    axes[0, 2].set_title("Joint-query to token attention")
    figure.suptitle("Wi-Pose mechanism evidence", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "attention_mechanisms.png"
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    np.savez_compressed(
        output_dir / "attention_mechanisms.npz",
        actions=np.asarray(labels),
        spatial_attention=spatial,
        temporal_attention=temporal,
        query_token_attention=query_token,
    )
    return output_path
