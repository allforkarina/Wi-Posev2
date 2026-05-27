from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torch.utils.data import DataLoader

from dataloader import create_da_data_loaders
from models import AXIAL_ENCODER_MODES, CECEModule, DECODER_TYPES, OPENPOSE_BONE_EDGES, WiFlowModel
from pose_targets import build_pcm_paf_targets


PCK_THRESHOLDS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
RIGHT_SHOULDER_INDEX = 2
LEFT_HIP_INDEX = 11


@dataclass(frozen=True)
class TrainConfig:
    dataset_root: str
    output_dir: str = "outputs/train"
    axial_mode: str = "spatial_then_temporal"
    decoder_type: str = "joint"
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-5
    max_lr: float = 1e-4
    weight_decay: float = 1e-3
    grad_clip_norm: float = 1.0
    bone_loss_weight: float = 0.5
    pct_start: float = 0.2
    heatmap_size: int = 36
    heatmap_sigma: float = 1.5
    paf_width: float = 1.0
    paf_loss_weight: float = 1.0
    num_workers: int = 8
    device: str = "cuda"
    seed: int = 42
    # DA fields
    source_envs: tuple[str, ...] = ("env1",)
    target_envs: tuple[str, ...] = ("env2",)
    alpha: float = 0.05
    ical_warmup_epochs: int = 10
    ical_sigma_pose: float = 1.0
    cece_enabled: bool = True
    # Regularization / training
    dropout: float = 0.1
    amp: bool = True
    early_stopping_patience: int = 15


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


def extract_prediction_keypoints(prediction: Any) -> torch.Tensor:
    if isinstance(prediction, Mapping):
        keypoints = prediction.get("keypoints")
        if not isinstance(keypoints, torch.Tensor):
            raise ValueError("Heatmap decoder output must contain tensor keypoints")
        return keypoints
    if not isinstance(prediction, torch.Tensor):
        raise TypeError(f"Unexpected model prediction type: {type(prediction)!r}")
    return prediction


def compute_losses(
    prediction: Any,
    target: torch.Tensor,
    bone_loss_weight: float = 0.5,
    heatmap_size: int = 36,
    heatmap_sigma: float = 1.5,
    paf_width: float = 1.0,
    paf_loss_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    zero = torch.zeros((), dtype=target.dtype, device=target.device)
    if isinstance(prediction, Mapping):
        stages = prediction.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("Heatmap decoder output must contain non-empty stages")
        pcm_gt, paf_gt = build_pcm_paf_targets(
            target,
            heatmap_size=heatmap_size,
            sigma=heatmap_sigma,
            paf_width=paf_width,
        )
        pcm_total = zero
        paf_total = zero
        for stage in stages:
            pcm_total = pcm_total + F.mse_loss(stage["pcm"], pcm_gt)
            paf_total = paf_total + F.mse_loss(stage["paf"], paf_gt)
        total = pcm_total + paf_loss_weight * paf_total
        return {
            "loss": total,
            "coord_loss": zero,
            "bone_loss": zero,
            "pcm_loss": pcm_total,
            "paf_loss": paf_total,
        }

    coord = F.l1_loss(prediction, target)
    bone = bone_length_loss(prediction, target)
    total = coord + bone_loss_weight * bone
    return {
        "loss": total,
        "coord_loss": coord,
        "bone_loss": bone,
        "pcm_loss": zero,
        "paf_loss": zero,
    }


def compute_ical_loss(
    f_s: torch.Tensor,
    f_t: torch.Tensor,
    y_s_gt: torch.Tensor,
    y_t_pred: torch.Tensor,
    sigma_pose: float = 1.0,
) -> torch.Tensor:
    """Instance-level consistency alignment loss.

    Reweights feature-space distances by pose similarity so that
    source-target pairs with similar poses are aligned more strongly.

    Args:
        f_s:       Source features after CECE reweighting + GAP, shape [B, D].
        f_t:       Target features after CECE reweighting + GAP, shape [B, D].
        y_s_gt:    Source ground-truth keypoints, shape [B, 18, 2].
        y_t_pred:  Target predicted keypoints, shape [B, 18, 2].
        sigma_pose: Temperature for pose-distance → similarity mapping.

    Returns:
        Scalar ICAL loss.
    """
    y_s_flat = y_s_gt.flatten(1)                            # [B, 36]
    y_t_flat = y_t_pred.flatten(1)                          # [B, 36]

    # Pairwise pose distances
    pose_dist = torch.cdist(y_s_flat, y_t_flat)             # [B, B]

    # Pose similarity weights with row-wise normalisation
    weights = torch.exp(-pose_dist / sigma_pose)            # [B, B]
    weights = weights / weights.sum(dim=1, keepdim=True)    # row-normalised

    # Weighted squared L2 feature distances
    f_dist_sq = torch.cdist(f_s, f_t, p=2).pow(2)          # [B, B]

    # Divide by B to keep loss magnitude stable across batch sizes
    return (weights * f_dist_sq).sum() / f_s.shape[0]


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


def run_da_epoch(
    model: nn.Module,
    cece: CECEModule | None,
    source_loader: DataLoader,
    target_loader: DataLoader,
    criterion_config: TrainConfig,
    device: torch.device,
    epoch: int,
    optimizer: AdamW | None = None,
    scheduler: LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Dict[str, float]:
    """Run one epoch of dual-domain training.

    Splits WiFlowModel's forward into spatial → CECE → decoder to
    compute per-domain supervised losses and the ICAL cross-domain loss.

    Source loader is iterated in a cycle; when exhausted the iterator
    is rebuilt to trigger a fresh shuffle (no fixed pairings across epochs).
    """
    is_training = optimizer is not None
    model.train(is_training)

    # ICAL warmup: linearly ramp alpha from 0 to config.alpha
    actual_alpha = criterion_config.alpha * min(
        1.0, epoch / max(criterion_config.ical_warmup_epochs, 1)
    )

    totals: Dict[str, float] = {}
    source_sample_count = 0
    target_sample_count = 0
    step_count = 0
    source_iter = iter(source_loader)
    use_amp = scaler is not None

    for batch_t in target_loader:
        # --- source batch (cycle with re-shuffle) ---
        try:
            batch_s = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            batch_s = next(source_iter)

        x_s, kp_s_gt = prepare_model_input(batch_s, device)
        x_t, kp_t_gt = prepare_model_input(batch_t, device)

        bs_s = x_s.shape[0]
        bs_t = x_t.shape[0]

        with torch.set_grad_enabled(is_training):
            with torch.amp.autocast(device.type, enabled=use_amp):
                # Forward: spatial → axial
                feat_s = model.axial_encoder(model.spatial_encoder(x_s))
                feat_t = model.axial_encoder(model.spatial_encoder(x_t))

                # CECE channel reweighting
                if criterion_config.cece_enabled and cece is not None:
                    feat_s_ce, feat_t_ce = cece(feat_s, feat_t)
                else:
                    feat_s_ce, feat_t_ce = feat_s, feat_t

                # Decode
                y_s = model.decode_features(feat_s_ce)
                y_t = model.decode_features(feat_t_ce)

                # Supervised losses
                losses_s = compute_losses(
                    y_s,
                    kp_s_gt,
                    bone_loss_weight=criterion_config.bone_loss_weight,
                )
                losses_t = compute_losses(
                    y_t,
                    kp_t_gt,
                    bone_loss_weight=criterion_config.bone_loss_weight,
                )

                # ICAL loss
                f_s_pooled = feat_s_ce.mean(dim=[2, 3])           # GAP → [B, 256]
                f_t_pooled = feat_t_ce.mean(dim=[2, 3])           # GAP → [B, 256]
                loss_ical = compute_ical_loss(
                    f_s_pooled, f_t_pooled, kp_s_gt, kp_t_gt,
                    sigma_pose=criterion_config.ical_sigma_pose,
                )

                loss = (losses_s["loss"] + losses_t["loss"]) / 2.0 + actual_alpha * loss_ical

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=criterion_config.grad_clip_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=criterion_config.grad_clip_norm,
                    )
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        # Metrics
        kp_s_pred = extract_prediction_keypoints(y_s).detach()
        kp_t_pred = extract_prediction_keypoints(y_t).detach()
        metrics_s = compute_metrics(kp_s_pred, kp_s_gt)
        metrics_t = compute_metrics(kp_t_pred, kp_t_gt)

        source_sample_count += bs_s
        target_sample_count += bs_t
        step_count += 1

        source_metric_items = {
            "source_loss": losses_s["loss"],
            "source_coord_loss": losses_s["coord_loss"],
            "source_bone_loss": losses_s["bone_loss"],
            "source_mpjpe": metrics_s["mpjpe"],
            "source_pck_0_2": metrics_s["pck_0_2"],
        }
        target_metric_items = {
            "target_loss": losses_t["loss"],
            "target_coord_loss": losses_t["coord_loss"],
            "target_bone_loss": losses_t["bone_loss"],
            "target_mpjpe": metrics_t["mpjpe"],
            "target_pck_0_2": metrics_t["pck_0_2"],
        }

        for name, value in {**source_metric_items, **target_metric_items}.items():
            weight = bs_s if name.startswith("source") else bs_t
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * weight
        totals["ical"] = totals.get("ical", 0.0) + float(loss_ical.detach().cpu())

    # Average
    averaged: Dict[str, float] = {}
    for name, total in totals.items():
        if name == "ical":
            averaged[name] = total / max(step_count, 1)
        else:
            count = source_sample_count if name.startswith("source") else target_sample_count
            averaged[name] = total / max(count, 1)
    return averaged


def run_val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion_config: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Run validation on the target domain only.

    Uses ``model.forward()`` directly — no CECE or ICAL applied.
    Note: training feeds CECE-reweighted features to the decoder, but
    validation feeds raw axial features.  This is intentional — at
    inference time no source-domain batch is available to compute CECE
    weights.  Val metrics may be slightly conservative because of this
    distribution gap.
    """
    model.eval()
    totals: Dict[str, float] = {}
    sample_count = 0

    with torch.no_grad():
        for batch in loader:
            model_input, target = prepare_model_input(batch, device)
            prediction = model(model_input)
            losses = compute_losses(
                prediction,
                target,
                bone_loss_weight=criterion_config.bone_loss_weight,
            )
            keypoint_prediction = extract_prediction_keypoints(prediction)
            metrics = compute_metrics(keypoint_prediction, target)

            bs = target.shape[0]
            sample_count += bs
            for name, value in {**losses, **metrics}.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * bs

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


def select_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def run_training(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    output_dir = Path(config.output_dir)

    loaders = create_da_data_loaders(
        data_dir=config.dataset_root,
        source_envs=config.source_envs,
        target_envs=config.target_envs,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    source_train_loader = loaders["source_train"]
    target_train_loader = loaders["target_train"]
    target_val_loader = loaders["target_val"]

    model = WiFlowModel(
        input_channels=3,
        axial_mode=config.axial_mode,
        decoder_type=config.decoder_type,
        heatmap_size=config.heatmap_size,
        dropout=config.dropout,
    ).to(device)

    cece = CECEModule(num_channels=256).to(device) if config.cece_enabled else None

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.max_lr,
        epochs=config.epochs,
        steps_per_epoch=len(target_train_loader),
        pct_start=config.pct_start,
        anneal_strategy="cos",
        div_factor=config.max_lr / max(config.lr, 1e-8),
        final_div_factor=1000.0,
    )

    scaler = torch.amp.GradScaler(device.type, enabled=config.amp and device.type == "cuda")

    # Sanity check
    first_batch = next(iter(target_train_loader))
    model_input, target = prepare_model_input(first_batch, device)
    with torch.no_grad():
        with torch.amp.autocast(device.type, enabled=scaler.is_enabled()):
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
    patience_counter = 0
    log_path = output_dir / "train_log.csv"
    for epoch in range(1, config.epochs + 1):
        start_time = time.perf_counter()
        train_metrics = run_da_epoch(
            model=model,
            cece=cece,
            source_loader=source_train_loader,
            target_loader=target_train_loader,
            criterion_config=config,
            device=device,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        val_metrics = run_val_epoch(model, target_val_loader, config, device)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - start_time

        actual_alpha = config.alpha * min(
            1.0, epoch / max(config.ical_warmup_epochs, 1)
        )

        row: Dict[str, float | int | str] = {
            "epoch": epoch,
            "axial_mode": config.axial_mode,
            "decoder_type": config.decoder_type,
            "train_source_loss": train_metrics["source_loss"],
            "train_source_coord_loss": train_metrics["source_coord_loss"],
            "train_source_bone_loss": train_metrics["source_bone_loss"],
            "train_source_mpjpe": train_metrics["source_mpjpe"],
            "train_source_pck_0_2": train_metrics["source_pck_0_2"],
            "train_target_loss": train_metrics["target_loss"],
            "train_target_coord_loss": train_metrics["target_coord_loss"],
            "train_target_bone_loss": train_metrics["target_bone_loss"],
            "train_target_mpjpe": train_metrics["target_mpjpe"],
            "train_target_pck_0_2": train_metrics["target_pck_0_2"],
            "train_ical": train_metrics["ical"],
            "alpha": actual_alpha,
            "val_loss": val_metrics["loss"],
            "val_coord_loss": val_metrics["coord_loss"],
            "val_bone_loss": val_metrics["bone_loss"],
            "val_pcm_loss": val_metrics["pcm_loss"],
            "val_paf_loss": val_metrics["paf_loss"],
            "val_mpjpe": val_metrics["mpjpe"],
            "val_pck_0_2": val_metrics["pck_0_2"],
            "val_pck_0_5": val_metrics["pck_0_5"],
            "heatmap_size": config.heatmap_size,
            "heatmap_sigma": config.heatmap_sigma,
            "paf_width": config.paf_width,
            "paf_loss_weight": config.paf_loss_weight,
            "current_lr": current_lr,
            "epoch_time": epoch_time,
        }
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
            patience_counter = 0
            save_checkpoint(
                output_dir / "best_val_mpjpe.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric=best_val_mpjpe,
                config=config,
            )
        else:
            patience_counter += 1
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
            f"src_loss={train_metrics['source_loss']:.6f} "
            f"tgt_loss={train_metrics['target_loss']:.6f} "
            f"ical={train_metrics['ical']:.6f} "
            f"val_mpjpe={val_metrics['mpjpe']:.6f} "
            f"val_pck_0_2={val_metrics['pck_0_2']:.6f} "
            f"lr={current_lr:.2e} "
            f"epoch_time={epoch_time:.1f}s"
        )

        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience_counter} epochs)")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the WiFlow pose model with domain adaptation.")
    parser.add_argument("--dataset-root", required=True, help="Path to the NPY memmap dataset directory.")
    parser.add_argument("--output-dir", default="outputs/train", help="Directory for logs and checkpoints.")
    parser.add_argument("--axial-mode", default="spatial_then_temporal", choices=AXIAL_ENCODER_MODES)
    parser.add_argument("--decoder-type", default="joint", choices=DECODER_TYPES)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--max-lr", type=float, default=1e-4, help="Peak learning rate for OneCycleLR.")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="AdamW weight decay.")
    parser.add_argument("--pct-start", type=float, default=0.2,
                        help="Fraction of training spent warming up LR in OneCycleLR.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for attention layers.")
    parser.add_argument("--no-amp", action="store_true", default=False,
                        help="Disable automatic mixed precision.")
    parser.add_argument("--early-stopping-patience", type=int, default=15,
                        help="Stop training after N epochs without val_mpjpe improvement.")
    # DA arguments
    parser.add_argument("--source-envs", nargs="+", default=["env1"],
                        help="Source domain environment names.")
    parser.add_argument("--target-envs", nargs="+", default=["env2"],
                        help="Target domain environment names.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="ICAL loss weight.")
    parser.add_argument("--ical-warmup-epochs", type=int, default=10,
                        help="Number of epochs to linearly ramp up ICAL alpha.")
    parser.add_argument("--ical-sigma-pose", type=float, default=1.0,
                        help="Temperature for ICAL pose-distance -> similarity mapping.")
    parser.add_argument("--no-cece", action="store_true", default=False,
                        help="Disable CECE channel reweighting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dict = vars(args)
    # Map CLI flags to config fields
    config_dict["cece_enabled"] = not config_dict.pop("no_cece")
    config_dict["amp"] = not config_dict.pop("no_amp")
    # Convert lists to tuples for frozen dataclass
    config_dict["source_envs"] = tuple(config_dict["source_envs"])
    config_dict["target_envs"] = tuple(config_dict["target_envs"])
    # Remove keys not in TrainConfig
    config = TrainConfig(**{
        k: v for k, v in config_dict.items()
        if k in TrainConfig.__dataclass_fields__
    })
    run_training(config)


if __name__ == "__main__":
    main()