from __future__ import annotations

import torch
from torch import nn

from data.pose_schema import NUM_KEYPOINTS


class WiFlowMLPDecoder(nn.Module):
    """Decode one globally pooled CSI feature vector into 18 joint coordinates."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding_dim = 256
        self.coordinate_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 1536),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1536, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, NUM_KEYPOINTS * 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.embedding_dim:
            raise ValueError("WiFlowMLPDecoder expects input shaped [B, 256, H, W]")
        if return_features:
            raise ValueError("MLP decoder does not expose joint latent features")
        pooled = x.mean(dim=(-2, -1))
        return self.coordinate_head(pooled).reshape(
            x.shape[0], NUM_KEYPOINTS, 2
        )
