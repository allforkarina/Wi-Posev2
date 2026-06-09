from __future__ import annotations

import torch
from torch import nn


CUSTOM_WRIST_JOINTS: tuple[int, int] = (10, 12)
WRIST_CONTEXT_JOINTS: tuple[int, int, int, int] = (8, 10, 12, 13)


class WristRefinementHead(nn.Module):
    """Apply a small residual correction to custom wrist joints 10 and 12."""

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        context_joints: tuple[int, ...] = WRIST_CONTEXT_JOINTS,
        refined_joints: tuple[int, ...] = CUSTOM_WRIST_JOINTS,
    ) -> None:
        super().__init__()
        self.context_joints = context_joints
        self.refined_joints = refined_joints
        self.input_layer = nn.Linear(len(context_joints) * embedding_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.output_layer = nn.Linear(hidden_dim, len(refined_joints) * 2)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        coordinates: torch.Tensor,
        decoder_features: torch.Tensor,
    ) -> torch.Tensor:
        if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
            raise ValueError("coordinates must be shaped [B, J, 2]")
        if decoder_features.ndim != 3:
            raise ValueError("decoder_features must be shaped [B, J, D]")
        if coordinates.shape[:2] != decoder_features.shape[:2]:
            raise ValueError("coordinates and decoder_features must share batch and joint dimensions")

        context_index = torch.as_tensor(
            self.context_joints,
            dtype=torch.long,
            device=decoder_features.device,
        )
        refine_index = torch.as_tensor(
            self.refined_joints,
            dtype=torch.long,
            device=coordinates.device,
        )
        context = decoder_features.index_select(dim=1, index=context_index)
        hidden = self.activation(self.input_layer(context.flatten(start_dim=1)))
        delta = self.output_layer(hidden).view(coordinates.shape[0], len(self.refined_joints), 2)

        refined = coordinates.clone()
        refined[:, refine_index] = refined[:, refine_index] + delta
        return refined
