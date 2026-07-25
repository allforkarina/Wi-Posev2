from __future__ import annotations

import torch
from torch import nn

from data.pose_schema import NUM_KEYPOINTS

from .skeleton import build_normalized_adjacency


class WiFlowHierarchicalJointDecoderStage(nn.Module):
    """One staged joint retrieval block with optional upstream joint context."""

    def __init__(self, has_context: bool, embedding_dim: int = 256) -> None:
        super().__init__()
        self.has_context = has_context
        self.embedding_dim = embedding_dim
        self.num_heads = 8
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.spatial_norm = nn.LayerNorm(embedding_dim)
        self.context_attention = (
            nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=self.num_heads,
                batch_first=True,
                dropout=0.1,
            )
            if has_context
            else None
        )
        self.context_norm = nn.LayerNorm(embedding_dim) if has_context else None
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.GELU(),
            nn.Linear(512, embedding_dim),
            nn.Dropout(0.1),
        )
        self.feedforward_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        query: torch.Tensor,
        spatial_tokens: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        spatial_output, _ = self.spatial_attention(
            query,
            spatial_tokens,
            spatial_tokens,
            need_weights=False,
        )
        h = self.spatial_norm(query + spatial_output)

        if self.context_attention is not None:
            if context is None:
                raise ValueError("context is required for this hierarchical decoder stage")
            context_output, _ = self.context_attention(h, context, context, need_weights=False)
            if self.context_norm is None:
                raise RuntimeError("context_norm is required when context_attention is enabled")
            h = self.context_norm(h + context_output)

        return self.feedforward_norm(h + self.feedforward(h))


class WiFlowHierarchicalJointDecoder(nn.Module):
    """Decode custom 18-joint coordinates through staged joint retrieval."""

    def __init__(self) -> None:
        super().__init__()
        self.num_queries = NUM_KEYPOINTS
        self.embedding_dim = 256
        self.num_heads = 4
        self.stage_indices = (
            (0, 3, 4, 6, 7),
            (15, 16, 9, 11, 14, 5, 13, 8),
            (17, 1, 2, 10, 12),
        )
        self.stage_order = tuple(joint for stage in self.stage_indices for joint in stage)
        self.openpose_order = tuple(self.stage_order.index(joint) for joint in range(self.num_queries))
        self.joint_queries = nn.Parameter(torch.zeros(self.num_queries, self.embedding_dim))
        self.stages = nn.ModuleList(
            WiFlowHierarchicalJointDecoderStage(
                has_context=stage_index > 0,
                embedding_dim=self.embedding_dim,
            )
            for stage_index in range(len(self.stage_indices))
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
            raise ValueError("WiFlowHierarchicalJointDecoder expects input shaped [B, 256, 29, 16]")
        return x.flatten(2).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        tokens = self.flatten_tokens(x)
        batch_size = tokens.shape[0]
        context_parts: list[torch.Tensor] = []
        stage_outputs: list[torch.Tensor] = []

        for stage, indices in zip(self.stages, self.stage_indices):
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=tokens.device)
            stage_query = self.joint_queries[index_tensor].unsqueeze(0).expand(batch_size, -1, -1)
            context = torch.cat(context_parts, dim=1) if context_parts else None
            stage_output = stage(stage_query, tokens, context)
            context_parts.append(stage_output)
            stage_outputs.append(stage_output)

        openpose_order = torch.as_tensor(self.openpose_order, dtype=torch.long, device=tokens.device)
        h = torch.cat(stage_outputs, dim=1)[:, openpose_order]
        gnn_output = torch.matmul(self.adjacency, self.gnn_projection(h))
        gnn_output = self.gnn_activation(gnn_output)
        h = self.gnn_norm(h + gnn_output)

        attention_output, _ = self.joint_attention(h, h, h, need_weights=False)
        h = self.attention_norm(h + attention_output)
        coordinates = self.coordinate_head(h)
        if return_features:
            return coordinates, h
        return coordinates
