from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torch.utils.data import DataLoader, Subset

from dataloader import create_few_shot_data_loader, create_memmap_data_loader, create_memmap_data_loaders
from models import (
    AXIAL_ENCODER_MODES,
    CSI_FEATURE_MODES,
    CSI_INPUT_CALIBRATION_TYPES,
    DECODER_TYPES,
    OPENPOSE_BONE_EDGES,
    SPATIAL_STEM_TYPES,
    WiFlowModel,
)


PCK_THRESHOLDS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
RIGHT_SHOULDER_INDEX = 2
LEFT_HIP_INDEX = 11
TRAINABLE_GROUPS: tuple[str, ...] = (
    "encoder",
    "decoder",
    "full",
    "input_calibration",
    "spatial_encoder",
    "axial_encoder",
    "axial_attention",
    "axial_projection",
    "decoder_queries",
    "decoder_attention",
    "decoder_head",
    "decoder_norms",
    "norms",
)
NORMALIZATION_MODULES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.GroupNorm,
    nn.LayerNorm,
)


@dataclass(frozen=True)
class TrainConfig:
    dataset_root: str
    output_dir: str = "outputs/train"
    axial_mode: str = "spatial_then_temporal"
    decoder_type: str = "joint"
    csi_feature_mode: str = "raw"
    spatial_stem_type: str = "baseline"
    background_kernel_size: int = 9
    input_calibration: str = "none"
    epochs: int = 50
    batch_size: int = 64
    lr: float = 2e-5
    max_lr: float = 5e-4
    weight_decay: float = 5e-4
    grad_clip_norm: float = 1.0
    bone_loss_weight: float = 0.5
    latent_structure_loss_weight: float = 0.0
    encoder_relation_loss_weight: float = 0.0
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42
    subset_size: int | None = None
    mode: str = ""
    source_envs: tuple[str, ...] | None = None
    target_envs: tuple[str, ...] | None = None
    finetune_from: str | None = None
    few_shot_subjects: int = 4
    few_shot_frames: int = 5
    trainable_groups: tuple[str, ...] = ("encoder",)
    freeze_tier: int | None = None


def prepare_model_input(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_input = torch.as_tensor(batch["csi_amplitude"], dtype=torch.float32, device=device)
    keypoints = torch.as_tensor(batch["keypoints"], dtype=torch.float32, device=device)
    return model_input, keypoints


def bone_length_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    edges: tuple[tuple[int, int], ...] = OPENPOSE_BONE_EDGES,
) -> torch.Tensor:
    edge_index = torch.as_tensor(edges, dtype=torch.long, device=prediction.device)
    pred_lengths = torch.linalg.vector_norm(
        prediction[:, edge_index[:, 0]] - prediction[:, edge_index[:, 1]],
        dim=-1,
    )
    target_lengths = torch.linalg.vector_norm(
        target[:, edge_index[:, 0]] - target[:, edge_index[:, 1]],
        dim=-1,
    )
    return F.l1_loss(pred_lengths, target_lengths)


def unpack_model_output(
    prediction: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(prediction, tuple):
        if len(prediction) != 2:
            raise ValueError(f"Unexpected model prediction tuple length: {len(prediction)}")
        keypoints, decoder_features = prediction
        if not isinstance(keypoints, torch.Tensor) or not isinstance(decoder_features, torch.Tensor):
            raise TypeError(f"Unexpected model prediction tuple types: {type(prediction)!r}")
        return keypoints, decoder_features
    if not isinstance(prediction, torch.Tensor):
        raise TypeError(f"Unexpected model prediction type: {type(prediction)!r}")
    return prediction, None


def extract_prediction_keypoints(
    prediction: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    prediction, _ = unpack_model_output(prediction)
    return prediction


def joint_latent_structure_loss(
    decoder_features: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if decoder_features.ndim != 3:
        raise ValueError("decoder_features must be shaped [B, J, D]")
    if target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError("target must be shaped [B, J, 2]")
    if decoder_features.shape[:2] != target.shape[:2]:
        raise ValueError(
            "decoder_features and target must share batch and joint dimensions, "
            f"got {tuple(decoder_features.shape)} and {tuple(target.shape)}"
        )

    normalized_features = F.normalize(decoder_features, dim=-1, eps=eps)
    latent_dist = 1.0 - torch.matmul(
        normalized_features,
        normalized_features.transpose(1, 2),
    )
    pose_dist = torch.cdist(target, target, p=2)

    num_joints = target.shape[1]
    off_diagonal = ~torch.eye(num_joints, dtype=torch.bool, device=target.device)
    latent_values = latent_dist[:, off_diagonal]
    pose_values = pose_dist[:, off_diagonal]
    latent_values = latent_values / latent_values.mean(dim=1, keepdim=True).clamp_min(eps)
    pose_values = pose_values / pose_values.mean(dim=1, keepdim=True).clamp_min(eps)
    return F.smooth_l1_loss(latent_values, pose_values)


def pose_feature_relation_loss(
    encoder_features: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if encoder_features.ndim < 2:
        raise ValueError("encoder_features must include batch and feature dimensions")
    if target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError("target must be shaped [B, J, 2]")
    if encoder_features.shape[0] != target.shape[0]:
        raise ValueError(
            "encoder_features and target must share batch dimension, "
            f"got {tuple(encoder_features.shape)} and {tuple(target.shape)}"
        )
    if target.shape[0] < 2:
        return encoder_features.sum() * 0.0

    if encoder_features.ndim == 4:
        feature_vector = encoder_features.mean(dim=(2, 3))
    else:
        feature_vector = encoder_features.flatten(1)
    feature_vector = F.normalize(feature_vector, dim=-1, eps=eps)
    feature_dist = 1.0 - torch.matmul(feature_vector, feature_vector.transpose(0, 1))

    pose_vector = target.flatten(1)
    pose_dist = torch.cdist(pose_vector, pose_vector, p=2)

    batch_size = target.shape[0]
    off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=target.device)
    feature_values = feature_dist[off_diagonal]
    pose_values = pose_dist[off_diagonal]
    feature_values = feature_values / feature_values.mean().clamp_min(eps)
    pose_values = pose_values / pose_values.mean().clamp_min(eps)
    return F.smooth_l1_loss(feature_values, pose_values)


def compute_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    bone_loss_weight: float = 0.5,
    decoder_features: torch.Tensor | None = None,
    latent_structure_loss_weight: float = 0.0,
    encoder_features: torch.Tensor | None = None,
    encoder_relation_loss_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    coord = F.l1_loss(prediction, target)
    bone = bone_length_loss(prediction, target)
    total = coord + bone_loss_weight * bone
    losses = {
        "loss": total,
        "coord_loss": coord,
        "bone_loss": bone,
    }
    if latent_structure_loss_weight > 0.0:
        if decoder_features is None:
            raise ValueError("decoder_features are required when latent_structure_loss_weight > 0")
        latent_structure = joint_latent_structure_loss(decoder_features, target)
        losses["latent_structure_loss"] = latent_structure
        losses["loss"] = losses["loss"] + latent_structure_loss_weight * latent_structure
    if encoder_relation_loss_weight > 0.0:
        if encoder_features is None:
            raise ValueError("encoder_features are required when encoder_relation_loss_weight > 0")
        encoder_relation = pose_feature_relation_loss(encoder_features, target)
        losses["encoder_relation_loss"] = encoder_relation
        losses["loss"] = losses["loss"] + encoder_relation_loss_weight * encoder_relation
    return losses


def compute_torso_scale(target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.linalg.vector_norm(
        target[:, RIGHT_SHOULDER_INDEX] - target[:, LEFT_HIP_INDEX],
        dim=-1,
    ).clamp_min(eps)


def mpjpe(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(prediction - target, dim=-1).mean()


def pck(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    errors = torch.linalg.vector_norm(prediction - target, dim=-1)
    scale = compute_torso_scale(target, eps=eps)
    return (errors < (scale[:, None] * threshold)).float().mean()


def compute_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    metrics = {"mpjpe": mpjpe(prediction, target)}
    for threshold in PCK_THRESHOLDS:
        metrics[f"pck_{threshold:.1f}".replace(".", "_")] = pck(prediction, target, threshold)
    return metrics


def average_meter_totals(totals: Dict[str, float], count: int) -> Dict[str, float]:
    return {name: value / max(count, 1) for name, value in totals.items()}


def run_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    criterion_config: TrainConfig,
    device: torch.device,
    optimizer: AdamW | None = None,
    scheduler: LRScheduler | None = None,
) -> Dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    totals: Dict[str, float] = {}
    sample_count = 0

    for batch in loader:
        model_input, target = prepare_model_input(batch, device)

        with torch.set_grad_enabled(is_training):
            encoder_features = None
            if criterion_config.encoder_relation_loss_weight > 0.0:
                encoder_features = model.encode_features(model_input)
                output = model.decode_features(
                    encoder_features,
                    return_decoder_features=criterion_config.latent_structure_loss_weight > 0.0,
                )
            else:
                output = model(
                    model_input,
                    return_decoder_features=criterion_config.latent_structure_loss_weight > 0.0,
                )
            prediction, decoder_features = unpack_model_output(output)
            losses = compute_losses(
                prediction,
                target,
                bone_loss_weight=criterion_config.bone_loss_weight,
                decoder_features=decoder_features,
                latent_structure_loss_weight=criterion_config.latent_structure_loss_weight,
                encoder_features=encoder_features,
                encoder_relation_loss_weight=criterion_config.encoder_relation_loss_weight,
            )
            keypoint_prediction = extract_prediction_keypoints(prediction)
            metrics = compute_metrics(keypoint_prediction.detach(), target)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=criterion_config.grad_clip_norm,
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        batch_size = target.shape[0]
        sample_count += batch_size
        for name, value in {**losses, **metrics}.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size

    return average_meter_totals(totals, sample_count)


def run_finetune_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    criterion_config: TrainConfig,
    device: torch.device,
    optimizer: AdamW,
    scheduler: LRScheduler | None = None,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    sample_count = 0

    for batch in loader:
        model_input, target = prepare_model_input(batch, device)
        optimizer.zero_grad(set_to_none=True)
        encoder_features = None
        if criterion_config.encoder_relation_loss_weight > 0.0:
            encoder_features = model.encode_features(model_input)
            output = model.decode_features(
                encoder_features,
                return_decoder_features=criterion_config.latent_structure_loss_weight > 0.0,
            )
        else:
            output = model(
                model_input,
                return_decoder_features=criterion_config.latent_structure_loss_weight > 0.0,
            )
        prediction, decoder_features = unpack_model_output(output)
        losses = compute_losses(
            prediction,
            target,
            bone_loss_weight=criterion_config.bone_loss_weight,
            decoder_features=decoder_features,
            latent_structure_loss_weight=criterion_config.latent_structure_loss_weight,
            encoder_features=encoder_features,
            encoder_relation_loss_weight=criterion_config.encoder_relation_loss_weight,
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=criterion_config.grad_clip_norm,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = target.shape[0]
        sample_count += batch_size
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size

    return average_meter_totals(totals, sample_count)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LRScheduler,
    epoch: int,
    best_metric: float,
    config: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "train_config": asdict(config),
        },
        path,
    )


def append_csv_row(path: Path, row: Mapping[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def maybe_subset_loader(loader: DataLoader, subset_size: int | None) -> DataLoader:
    if subset_size is None:
        return loader
    subset_indices = list(range(min(subset_size, len(loader.dataset))))
    return DataLoader(
        Subset(loader.dataset, subset_indices),
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
    )


def select_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _normalization_parameter_names(model: nn.Module) -> set[str]:
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, NORMALIZATION_MODULES):
            continue
        for param_name, _ in module.named_parameters(recurse=False):
            names.add(f"{module_name}.{param_name}" if module_name else param_name)
    return names


def _matches_trainable_group(
    name: str,
    group: str,
    norm_param_names: set[str],
) -> bool:
    if group == "encoder":
        return name.startswith(("spatial_encoder.", "axial_encoder."))
    if group == "decoder":
        return name.startswith("decoder.")
    if group == "full":
        return True
    if group == "input_calibration":
        return name.startswith("input_calibration.")
    if group == "spatial_encoder":
        return name.startswith("spatial_encoder.")
    if group == "axial_encoder":
        return name.startswith("axial_encoder.")
    if group == "axial_attention":
        return name.startswith((
            "axial_encoder.spatial_attention.",
            "axial_encoder.temporal_attention.",
            "axial_encoder.spatial_norm.",
            "axial_encoder.temporal_norm.",
        ))
    if group == "axial_projection":
        return name.startswith((
            "axial_encoder.channel_projection.",
            "axial_encoder.concat_projection.",
        ))
    if group == "decoder_queries":
        return name.startswith("decoder.joint_queries")
    if group == "decoder_attention":
        return name.startswith((
            "decoder.cross_attention_layers.",
            "decoder.stages.",
            "decoder.joint_attention.",
        ))
    if group == "decoder_head":
        return name.startswith("decoder.coordinate_head.")
    if group == "decoder_norms":
        return name.startswith("decoder.") and name in norm_param_names
    if group == "norms":
        return name in norm_param_names
    raise ValueError(f"Unknown trainable group: {group}")


def apply_trainable_groups(model: nn.Module, groups: tuple[str, ...]) -> int:
    if not groups:
        raise ValueError(
            f"At least one trainable group is required. Valid groups: {TRAINABLE_GROUPS}"
        )

    unknown = tuple(group for group in groups if group not in TRAINABLE_GROUPS)
    if unknown:
        raise ValueError(
            f"Unknown trainable group(s): {unknown}. Valid groups: {TRAINABLE_GROUPS}"
        )

    selected = set(groups)
    norm_param_names = _normalization_parameter_names(model)
    trainable_params = 0
    for name, param in model.named_parameters():
        keep = any(
            _matches_trainable_group(name, group, norm_param_names)
            for group in selected
        )
        param.requires_grad = keep
        if keep:
            trainable_params += param.numel()

    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable groups {tuple(groups)}: {trainable_params}/{total} parameters trainable "
        f"({trainable_params / total * 100:.1f}%)"
    )
    return trainable_params


def load_finetune_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> torch.nn.modules.module._IncompatibleKeys:
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        name
        for name, _ in model.named_parameters()
        if name.startswith("input_calibration.")
    }
    unexpected = list(incompatible.unexpected_keys)
    missing = [name for name in incompatible.missing_keys if name not in allowed_missing]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the requested model architecture. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}"
        )
    if incompatible.missing_keys:
        print(
            "Initialized new input calibration parameters: "
            f"{tuple(incompatible.missing_keys)}"
        )
    return incompatible


def apply_finetune_tier(model: nn.Module, tier: int = 1) -> int:
    trainable_params = 0
    if tier == 1:
        keep_kw = ("norm", "bn", "ln", "joint_queries", "coordinate_head")
    elif tier == 2:
        keep_kw = ("norm", "bn", "ln", "joint_queries", "coordinate_head", "decoder.")
    elif tier == 3:
        keep_kw = ("norm", "bn", "ln", "joint_queries", "coordinate_head", "decoder.",
                   "spatial_attention.")
    elif tier == 4:
        keep_kw = ("norm", "bn", "ln", "joint_queries", "coordinate_head", "decoder.",
                   "spatial_attention.", "temporal_attention.")
    elif tier == 5:
        keep_kw = ("norm", "bn", "ln", "joint_queries", "coordinate_head", "decoder.",
                   "spatial_attention.", "temporal_attention.", "channel_projection")
    elif tier == 6:
        for param in model.parameters():
            param.requires_grad = True
            trainable_params += param.numel()
        total = sum(p.numel() for p in model.parameters())
        print(f"Freeze tier {tier}: {trainable_params}/{total} parameters trainable "
              f"({trainable_params / total * 100:.1f}%)")
        return trainable_params
    else:
        raise ValueError(f"Unknown freeze tier: {tier}. Valid tiers: 1-6")

    for name, param in model.named_parameters():
        keep = any(kw in name.lower() for kw in keep_kw)
        param.requires_grad = keep
        if keep:
            trainable_params += param.numel()

    total = sum(p.numel() for p in model.parameters())
    print(f"Freeze tier {tier}: {trainable_params}/{total} parameters trainable "
          f"({trainable_params / total * 100:.1f}%)")
    return trainable_params


def _run_source_only(config: TrainConfig, device: torch.device, output_dir: Path) -> None:
    envs = config.source_envs if config.source_envs else None
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        loaders[split] = create_memmap_data_loader(
            data_dir=config.dataset_root,
            split=split,
            batch_size=config.batch_size,
            envs=envs,
            num_workers=config.num_workers,
            seed=config.seed,
        )

    train_loader = maybe_subset_loader(loaders["train"], config.subset_size)
    val_loader = maybe_subset_loader(loaders["val"], config.subset_size)
    model = WiFlowModel(
        input_channels=3,
        axial_mode=config.axial_mode,
        decoder_type=config.decoder_type,
        csi_feature_mode=config.csi_feature_mode,
        spatial_stem_type=config.spatial_stem_type,
        background_kernel_size=config.background_kernel_size,
        input_calibration=config.input_calibration,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.max_lr,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=config.max_lr / max(config.lr, 1e-8),
        final_div_factor=1000.0,
    )

    first_batch = next(iter(train_loader))
    model_input, target = prepare_model_input(first_batch, device)
    with torch.no_grad():
        output = model(model_input)
    keypoint_output = extract_prediction_keypoints(output)
    print(
        "Sanity shapes: "
        f"input={tuple(model_input.shape)}, output={tuple(keypoint_output.shape)}, label={tuple(target.shape)}"
    )
    if keypoint_output.shape != target.shape:
        raise ValueError(
            f"Model output shape {tuple(keypoint_output.shape)} does not match label shape {tuple(target.shape)}"
        )

    best_val_mpjpe = float("inf")
    best_val_pck_0_2 = -float("inf")
    log_path = output_dir / "train_log.csv"
    for epoch in range(1, config.epochs + 1):
        start_time = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            config,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        val_metrics = run_epoch(model, val_loader, config, device)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - start_time

        row: Dict[str, float | int | str] = {
            "epoch": epoch,
            "axial_mode": config.axial_mode,
            "decoder_type": config.decoder_type,
            "csi_feature_mode": config.csi_feature_mode,
            "spatial_stem_type": config.spatial_stem_type,
            "background_kernel_size": config.background_kernel_size,
            "input_calibration": config.input_calibration,
            "latent_structure_loss_weight": config.latent_structure_loss_weight,
            "encoder_relation_loss_weight": config.encoder_relation_loss_weight,
            "train_loss": train_metrics["loss"],
            "train_coord_loss": train_metrics["coord_loss"],
            "train_bone_loss": train_metrics["bone_loss"],
            "train_mpjpe": train_metrics["mpjpe"],
            "train_pck_0_2": train_metrics["pck_0_2"],
            "val_loss": val_metrics["loss"],
            "val_coord_loss": val_metrics["coord_loss"],
            "val_bone_loss": val_metrics["bone_loss"],
            "val_mpjpe": val_metrics["mpjpe"],
            "val_pck_0_2": val_metrics["pck_0_2"],
            "val_pck_0_5": val_metrics["pck_0_5"],
            "current_lr": current_lr,
            "epoch_time": epoch_time,
        }
        if "latent_structure_loss" in train_metrics:
            row["train_latent_structure_loss"] = train_metrics["latent_structure_loss"]
        if "latent_structure_loss" in val_metrics:
            row["val_latent_structure_loss"] = val_metrics["latent_structure_loss"]
        if "encoder_relation_loss" in train_metrics:
            row["train_encoder_relation_loss"] = train_metrics["encoder_relation_loss"]
        if "encoder_relation_loss" in val_metrics:
            row["val_encoder_relation_loss"] = val_metrics["encoder_relation_loss"]
        append_csv_row(log_path, row)

        save_checkpoint(
            output_dir / "last.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_metric=val_metrics["mpjpe"],
            config=config,
        )
        if val_metrics["mpjpe"] < best_val_mpjpe:
            best_val_mpjpe = val_metrics["mpjpe"]
            save_checkpoint(
                output_dir / "best_val_mpjpe.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric=best_val_mpjpe,
                config=config,
            )
        if val_metrics["pck_0_2"] > best_val_pck_0_2:
            best_val_pck_0_2 = val_metrics["pck_0_2"]
            save_checkpoint(
                output_dir / "best_val_pck_0_2.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric=best_val_pck_0_2,
                config=config,
            )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_mpjpe={val_metrics['mpjpe']:.6f} "
            f"val_pck_0_2={val_metrics['pck_0_2']:.6f} "
            f"lr={current_lr:.2e} "
            f"epoch_time={epoch_time:.1f}s"
        )


def _run_finetune(config: TrainConfig, device: torch.device, output_dir: Path) -> None:
    target_envs = config.target_envs
    if not target_envs:
        raise ValueError("--target-envs required for finetune mode")

    train_loader, val_loader, train_indices = create_few_shot_data_loader(
        data_dir=config.dataset_root,
        target_envs=target_envs,
        few_shot_subjects=config.few_shot_subjects,
        few_shot_frames=config.few_shot_frames,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    print(f"Few-shot train: {len(train_indices)} frames, val: {len(val_loader.dataset)} frames")

    indices_path = output_dir / "few_shot_train_indices.npy"
    indices_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(indices_path), np.array(train_indices, dtype=np.int64))

    if not config.finetune_from:
        raise ValueError("--finetune-from required for finetune mode")
    checkpoint = torch.load(config.finetune_from, map_location=device)
    model = WiFlowModel(
        input_channels=3,
        axial_mode=config.axial_mode,
        decoder_type=config.decoder_type,
        csi_feature_mode=config.csi_feature_mode,
        spatial_stem_type=config.spatial_stem_type,
        background_kernel_size=config.background_kernel_size,
        input_calibration=config.input_calibration,
    ).to(device)
    if "model_state_dict" in checkpoint:
        load_finetune_state_dict(model, checkpoint["model_state_dict"])
    else:
        load_finetune_state_dict(model, checkpoint)
    if config.freeze_tier is not None:
        apply_finetune_tier(model, tier=config.freeze_tier)
    else:
        apply_trainable_groups(model, groups=config.trainable_groups)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.max_lr,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=config.max_lr / max(config.lr, 1e-8),
        final_div_factor=1000.0,
    )

    first_batch = next(iter(train_loader))
    model_input, target = prepare_model_input(first_batch, device)
    with torch.no_grad():
        output = model(model_input)
    print(
        "Sanity shapes: "
        f"input={tuple(model_input.shape)}, output={tuple(output.shape)}, label={tuple(target.shape)}"
    )

    best_train_loss = float("inf")
    log_path = output_dir / "train_log.csv"
    for epoch in range(1, config.epochs + 1):
        start_time = time.perf_counter()
        train_metrics = run_finetune_epoch(
            model, train_loader, config, device, optimizer, scheduler
        )
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - start_time

        row: Dict[str, float | int | str] = {
            "epoch": epoch,
            "csi_feature_mode": config.csi_feature_mode,
            "spatial_stem_type": config.spatial_stem_type,
            "background_kernel_size": config.background_kernel_size,
            "input_calibration": config.input_calibration,
            "latent_structure_loss_weight": config.latent_structure_loss_weight,
            "encoder_relation_loss_weight": config.encoder_relation_loss_weight,
            "train_loss": train_metrics["loss"],
            "train_coord_loss": train_metrics["coord_loss"],
            "train_bone_loss": train_metrics["bone_loss"],
            "current_lr": current_lr,
            "epoch_time": epoch_time,
        }
        if "latent_structure_loss" in train_metrics:
            row["train_latent_structure_loss"] = train_metrics["latent_structure_loss"]
        if "encoder_relation_loss" in train_metrics:
            row["train_encoder_relation_loss"] = train_metrics["encoder_relation_loss"]
        append_csv_row(log_path, row)

        save_checkpoint(
            output_dir / f"epoch_{epoch:03d}.pth",
            model, optimizer, scheduler, epoch,
            best_metric=train_metrics["loss"],
            config=config,
        )
        if train_metrics["loss"] < best_train_loss:
            best_train_loss = train_metrics["loss"]
            save_checkpoint(
                output_dir / "best_train_loss.pth",
                model, optimizer, scheduler, epoch,
                best_metric=best_train_loss,
                config=config,
            )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"lr={current_lr:.2e} "
            f"epoch_time={epoch_time:.1f}s"
        )


def run_training(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    output_dir = Path(config.output_dir)

    if config.mode == "source_only":
        _run_source_only(config, device, output_dir)
    elif config.mode == "finetune":
        _run_finetune(config, device, output_dir)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the WiFlow pose model.")
    parser.add_argument("--mode", required=True, choices=("source_only", "finetune"))
    parser.add_argument("--dataset-root", required=True, help="Path to the NPY memmap dataset directory.")
    parser.add_argument("--output-dir", default="outputs/train", help="Directory for logs and checkpoints.")
    parser.add_argument("--axial-mode", default="spatial_then_temporal", choices=AXIAL_ENCODER_MODES)
    parser.add_argument("--decoder-type", default="joint", choices=DECODER_TYPES)
    parser.add_argument(
        "--csi-feature-mode",
        default="raw",
        choices=CSI_FEATURE_MODES,
        help=(
            "CSI input representation before the spatial encoder. "
            "raw preserves the 3-channel baseline; physics_bank expands to 12 channels."
        ),
    )
    parser.add_argument(
        "--spatial-stem",
        dest="spatial_stem_type",
        default="baseline",
        choices=SPATIAL_STEM_TYPES,
        help=(
            "Spatial encoder stem. baseline preserves the release2.0 front-end; "
            "background_gated separates slow temporal CSI background from residual perturbations."
        ),
    )
    parser.add_argument(
        "--background-kernel-size",
        type=int,
        default=9,
        help="Odd temporal kernel size for the background_gated low-pass branch.",
    )
    parser.add_argument(
        "--input-calibration",
        default="none",
        choices=CSI_INPUT_CALIBRATION_TYPES,
        help=(
            "Optional CSI input calibration before feature-bank expansion. "
            "antenna_subcarrier_affine learns per-antenna/subcarrier scale and bias."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--latent-structure-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for decoder joint-latent pairwise structure loss. "
            "Default 0 preserves the baseline coordinate and bone losses."
        ),
    )
    parser.add_argument(
        "--encoder-relation-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for target-domain encoder feature relation loss. "
            "Default 0 preserves the baseline coordinate and bone losses."
        ),
    )
    parser.add_argument("--source-envs", nargs="*", default=None, help="Source environment names")
    parser.add_argument("--target-envs", nargs="*", default=None, help="Target environment names")
    parser.add_argument("--finetune-from", default=None, help="Path to source checkpoint for finetune")
    parser.add_argument("--few-shot-subjects", type=int, default=4)
    parser.add_argument("--few-shot-frames", type=int, default=5)
    parser.add_argument(
        "--trainable-groups",
        nargs="+",
        choices=TRAINABLE_GROUPS,
        default=("encoder",),
        help=(
            "Parameter groups to train during finetune. Main ablations: encoder, "
            "decoder, full. Fine-grained groups include spatial_encoder, "
            "axial_encoder, axial_attention, axial_projection, decoder_queries, "
            "decoder_attention, decoder_head, decoder_norms, input_calibration, norms."
        ),
    )
    parser.add_argument(
        "--freeze-tier",
        type=int,
        default=None,
        help=(
            "Deprecated compatibility option: cumulative freeze tier "
            "1(norms+head) 2(+decoder) 3(+spatial_attn) 4(+temporal_attn) "
            "5(+channel_proj) 6(full). Prefer --trainable-groups for ablations."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dict = vars(args)
    if config_dict.get("source_envs") is not None:
        config_dict["source_envs"] = tuple(config_dict["source_envs"])
    if config_dict.get("target_envs") is not None:
        config_dict["target_envs"] = tuple(config_dict["target_envs"])
    if config_dict.get("trainable_groups") is not None:
        config_dict["trainable_groups"] = tuple(config_dict["trainable_groups"])
    field_names = {f.name for f in TrainConfig.__dataclass_fields__.values()}
    config = TrainConfig(**{k: v for k, v in config_dict.items() if k in field_names})
    run_training(config)


if __name__ == "__main__":
    main()
