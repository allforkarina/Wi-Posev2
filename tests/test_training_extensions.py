from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from eval import _compute_diagnostics
from train import (
    LOWER_LIMB_JOINTS,
    TrainConfig,
    coral_loss,
    run_finetune_epoch,
    weighted_coordinate_loss,
)


def test_uniform_coordinate_loss_preserves_original_l1() -> None:
    prediction = torch.randn(2, 18, 2)
    target = torch.randn(2, 18, 2)

    actual = weighted_coordinate_loss(
        prediction,
        target,
        joint_loss_preset="uniform",
        lower_limb_weight=2.0,
    )

    assert torch.allclose(actual, F.l1_loss(prediction, target))


def test_lower_limb_loss_uses_custom_schema_hips_knees_and_ankles() -> None:
    assert LOWER_LIMB_JOINTS == (16, 5, 2, 15, 14, 17)
    target = torch.zeros(1, 18, 2)
    lower_limb_error = target.clone()
    lower_limb_error[:, 17] = 1.0
    arm_error = target.clone()
    arm_error[:, 10] = 1.0

    lower_limb_loss = weighted_coordinate_loss(
        lower_limb_error,
        target,
        joint_loss_preset="lower_limb",
        lower_limb_weight=2.0,
    )
    arm_loss = weighted_coordinate_loss(
        arm_error,
        target,
        joint_loss_preset="lower_limb",
        lower_limb_weight=2.0,
    )

    assert lower_limb_loss > arm_loss


def test_coral_loss_detects_covariance_shift() -> None:
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    target = source.clone()
    target[:, 0] *= 3.0

    assert torch.isclose(coral_loss(source, source.clone()), torch.tensor(0.0))
    assert coral_loss(source, target) > 0.0


class _TinyAlignmentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        self.decoder = nn.Linear(2, 36, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(2))

    def encode_features(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)

    def decode_features(
        self,
        features: torch.Tensor,
        return_decoder_features: bool = False,
    ) -> torch.Tensor:
        if return_decoder_features:
            raise AssertionError("This test does not request decoder features")
        return self.decoder(features).view(features.shape[0], 18, 2)

    def forward(
        self,
        values: torch.Tensor,
        return_decoder_features: bool = False,
    ) -> torch.Tensor:
        return self.decode_features(
            self.encode_features(values),
            return_decoder_features=return_decoder_features,
        )


def _tiny_batch(values: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "csi_amplitude": values,
        "keypoints": torch.zeros(values.shape[0], 18, 2),
    }


def test_alignment_finetune_logs_matched_source_and_coral_terms() -> None:
    model = _TinyAlignmentModel()
    source_loader = [
        _tiny_batch(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
    ]
    target_loader = [
        _tiny_batch(torch.tensor([[3.0, 0.0], [0.0, 1.0], [-3.0, 0.0]]))
    ]
    config = TrainConfig(
        dataset_root="unused",
        mode="finetune_align",
        align_loss="coral",
        align_weight=0.25,
        bone_loss_weight=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = run_finetune_epoch(
        model,
        target_loader,
        config,
        torch.device("cpu"),
        optimizer,
        source_loader=source_loader,
    )

    assert metrics["source_loss"] >= 0.0
    assert metrics["target_loss"] >= 0.0
    assert metrics["align_loss"] > 0.0
    assert metrics["loss"] >= metrics["target_loss"]


def test_diagnostics_use_custom_joint_names_groups_and_std() -> None:
    targets = np.zeros((3, 18, 2), dtype=np.float32)
    predictions = np.zeros((3, 18, 2), dtype=np.float32)
    targets[:, 17, 0] = np.array([0.0, 2.0, 4.0], dtype=np.float32)
    predictions[:, 17, 0] = np.array([1.0, 1.5, 2.0], dtype=np.float32)

    diagnostic = _compute_diagnostics([predictions], [targets])
    ankle = diagnostic["joint_rows"][17]

    assert ankle["joint_name"] == "right_ankle"
    assert "right_leg" in ankle["joint_groups"]
    assert "lower_limb" in ankle["joint_groups"]
    assert np.isclose(ankle["gt_std"], np.sqrt(ankle["gt_var"]))
    assert np.isclose(ankle["std_ratio"], ankle["pred_std"] / ankle["gt_std"])
    assert diagnostic["overall"]["overall_std_ratio"] > 0.0
