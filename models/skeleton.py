from __future__ import annotations

import torch

from data.pose_schema import CANONICAL_BONE_EDGES, NUM_KEYPOINTS


# Backward-compatible aliases.  The ordering is project-specific, not OpenPose.
NUM_OPENPOSE_KEYPOINTS = NUM_KEYPOINTS
OPENPOSE_BONE_EDGES = CANONICAL_BONE_EDGES


def build_normalized_adjacency(
    num_nodes: int = NUM_OPENPOSE_KEYPOINTS,
    edges: tuple[tuple[int, int], ...] = OPENPOSE_BONE_EDGES,
) -> torch.Tensor:
    """Build symmetric normalized adjacency with self-loops."""

    adjacency = torch.eye(num_nodes, dtype=torch.float32)
    for start, end in edges:
        adjacency[start, end] = 1.0
        adjacency[end, start] = 1.0

    degree = adjacency.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]


def build_decoder_adjacency(variant: str = "canonical") -> torch.Tensor:
    """Return a matched-control adjacency for decoder ablations."""
    canonical = build_normalized_adjacency()
    if variant == "canonical":
        return canonical
    if variant == "identity":
        return torch.eye(NUM_OPENPOSE_KEYPOINTS, dtype=torch.float32)
    if variant == "shuffled":
        generator = torch.Generator().manual_seed(3407)
        permutation = torch.randperm(
            NUM_OPENPOSE_KEYPOINTS,
            generator=generator,
        )
        return canonical.index_select(0, permutation).index_select(1, permutation)
    raise ValueError(
        "Decoder adjacency variant must be canonical, identity, or shuffled"
    )
