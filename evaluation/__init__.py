"""Evaluation utilities for WiFlow model analysis.

This package contains tools for model evaluation and research-grade
feature visualization.
"""

from evaluation.hooks import WiFlowHookContext, wiflow_hooks
from evaluation.feature_viz import run_feature_visualization

__all__ = [
    "WiFlowHookContext",
    "wiflow_hooks",
    "run_feature_visualization",
]