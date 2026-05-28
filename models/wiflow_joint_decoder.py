from __future__ import annotations

import torch
from torch import nn

from .skeleton import NUM_OPENPOSE_KEYPOINTS, build_normalized_adjacency


class WiFlowJointCrossAttentionLayer(nn.Module):
    """One refinement step for joint queries attending to spatial tokens."""

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = 8
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.cross_attention_norm = nn.LayerNorm(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.GELU(),
            nn.Linear(512, embedding_dim),
            nn.Dropout(0.1),
        )
        self.feedforward_norm = nn.LayerNorm(embedding_dim)

    def forward(self, joint_queries: torch.Tensor, spatial_tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.cross_attention(
            joint_queries,
            spatial_tokens,
            spatial_tokens,
            need_weights=False,
        )
        h = self.cross_attention_norm(joint_queries + attended)
        return self.feedforward_norm(h + self.feedforward(h))


class WiFlowJointDecoder(nn.Module):
    """Decode OpenPose18 coordinates from [B, 256, 29, 16] feature maps."""

    def __init__(self, num_layers: int = 3) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.num_queries = NUM_OPENPOSE_KEYPOINTS
        self.embedding_dim = 256
        self.num_layers = num_layers
        self.num_heads = 4
        self.joint_queries = nn.Parameter(torch.zeros(self.num_queries, self.embedding_dim))
        self.cross_attention_layers = nn.ModuleList(
            WiFlowJointCrossAttentionLayer(embedding_dim=self.embedding_dim)
            for _ in range(num_layers)
        )
        self.gnn_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.gnn_activation = nn.SiLU()
        self.gnn_norm = nn.LayerNorm(self.embedding_dim)
        self.joint_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.attention_norm = nn.LayerNorm(self.embedding_dim)
        self.coordinate_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 2),
        )
        self.register_buffer("adjacency", build_normalized_adjacency(), persistent=False)

    def flatten_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("WiFlowJointDecoder expects input shaped [B, 256, 29, 16]")
        return x.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.flatten_tokens(x)
        h = self.joint_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for layer in self.cross_attention_layers:
            h = layer(h, tokens)

        gnn_output = torch.matmul(self.adjacency, self.gnn_projection(h))
        gnn_output = self.gnn_activation(gnn_output)
        h = self.gnn_norm(h + gnn_output)

        attention_output, _ = self.joint_attention(h, h, h, need_weights=False)
        h = self.attention_norm(h + attention_output)
        return self.coordinate_head(h)
