from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import (  # noqa: E402
    DISTAL_SUPERVISION_JOINTS,
    TrainConfig,
    build_distal_hard_replay_subset,
    distal_joint_loss,
    run_finetune_epoch,
)


class TinyPoseDataset(torch.utils.data.Dataset):
    def __init__(self, distal_values: list[float]) -> None:
        self.keypoints: list[torch.Tensor] = []
        for value in distal_values:
            pose = torch.zeros(18, 2)
            pose[list(DISTAL_SUPERVISION_JOINTS), 0] = value
            self.keypoints.append(pose)

    def __len__(self) -> int:
        return len(self.keypoints)

    def __getitem__(self, index: int) -> dict:
        return {
            "csi": torch.zeros(64, 3, 114),
            "kpts18": self.keypoints[index],
            "meta": {
                "env": "env2",
                "subject": "S01",
                "action": "A01",
                "frame_idx": index,
            },
        }


class TinyPoseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, return_decoder_features: bool = False) -> torch.Tensor:
        return self.offset.expand(x.shape[0], 18, 2)


def _batch(target_value: float, batch_size: int = 2) -> dict:
    keypoints = torch.zeros(batch_size, 18, 2)
    keypoints[:, list(DISTAL_SUPERVISION_JOINTS), 0] = target_value
    return {
        "csi_amplitude": torch.zeros(batch_size, 3, 114, 64),
        "keypoints": keypoints,
    }


def test_distal_joint_loss_ignores_non_distal_coordinates() -> None:
    prediction = torch.zeros(1, 18, 2)
    target = torch.zeros(1, 18, 2)
    target[:, :, 0] = 10.0
    target[:, list(DISTAL_SUPERVISION_JOINTS), 0] = 1.0

    loss = distal_joint_loss(prediction, target)

    assert loss == pytest.approx(torch.tensor(1.0))


def test_distal_hard_replay_repeats_high_distal_variation_frames() -> None:
    dataset = TinyPoseDataset([0.0, 0.1, 4.0, 0.2])

    replay_dataset = build_distal_hard_replay_subset(
        dataset,
        replay_factor=3,
        replay_fraction=0.25,
    )

    assert len(replay_dataset) == 6
    replay_indices = list(replay_dataset.indices)
    assert replay_indices.count(2) == 3
    assert replay_indices.count(0) == 1
    assert replay_indices.count(1) == 1
    assert replay_indices.count(3) == 1


def test_finetune_epoch_logs_distal_and_source_replay_losses() -> None:
    model = TinyPoseModel()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    config = TrainConfig(
        dataset_root="unused",
        mode="finetune",
        distal_loss_weight=2.0,
        source_replay_weight=0.5,
    )
    target_loader = [_batch(target_value=1.0)]
    source_loader = [_batch(target_value=0.25)]

    metrics = run_finetune_epoch(
        model,
        target_loader,
        config,
        torch.device("cpu"),
        optimizer,
        source_loader=source_loader,
    )

    assert "target_loss" in metrics
    assert "source_loss" in metrics
    assert "distal_loss" in metrics
    assert metrics["loss"] > metrics["target_loss"]
