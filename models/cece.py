from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CECEModule(nn.Module):
    """Cross-Environment Channel Enhancement.

    Computes per-channel cosine similarity between source and target domain
    feature maps, then reweights both domains' features by channel consistency
    scores.  Stateless — no learnable parameters.  Only used during training.
    """

    def __init__(self, num_channels: int = 256) -> None:
        super().__init__()
        self.num_channels = num_channels

    def forward(
        self,
        src_feat: torch.Tensor,
        tgt_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # src_feat, tgt_feat: [B, C, H, W]
        C = src_feat.shape[1]

        # Batch-average to obtain domain-level representative feature maps
        src_mean = src_feat.mean(dim=0)          # [C, H, W]
        tgt_mean = tgt_feat.mean(dim=0)          # [C, H, W]

        # Flatten spatial dims: each channel becomes a vector in R^{H*W}
        src_flat = src_mean.view(C, -1)          # [C, H*W]
        tgt_flat = tgt_mean.view(C, -1)          # [C, H*W]

        # Per-channel cosine similarity → [C], range [-1, 1]
        cos_sim = F.cosine_similarity(src_flat, tgt_flat, dim=1)

        # Linear map to [0, 1]; channels with negative similarity get weight < 0.5
        weights = (cos_sim + 1.0) / 2.0          # [C]
        weights = weights.view(1, C, 1, 1)       # broadcast shape

        return src_feat * weights, tgt_feat * weights