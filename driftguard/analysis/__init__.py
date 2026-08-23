"""Temporal redundancy analysis utilities."""

from .feature_blocks import (
    RollingFeatureBuffer,
    block_metrics,
    split_feature_blocks,
)
from .temporal_similarity import (
    analyze_feature_stream,
    cosine_similarity,
    relative_l2_distance,
)

__all__ = [
    "RollingFeatureBuffer",
    "analyze_feature_stream",
    "block_metrics",
    "cosine_similarity",
    "relative_l2_distance",
    "split_feature_blocks",
]
