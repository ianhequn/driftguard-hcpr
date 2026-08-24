"""Shared cache-state semantics for refresh-policy experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from driftguard.analysis.cheap_observables import replace_blocks


def validate_analysis_only_policies(
    deployable_policies: tuple[str, ...],
    analysis_only_policies: tuple[str, ...] = ("oracle",),
) -> None:
    overlap = set(deployable_policies) & set(analysis_only_policies)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"analysis-only policies marked deployable: {names}")


def refresh_count(block_count: int, ratio: float) -> int:
    if block_count <= 0:
        raise ValueError("block_count must be positive")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be between 0 and 1")
    return min(block_count, int(np.ceil(block_count * ratio)))


def select_top_k(scores: np.ndarray, ratio: float) -> np.ndarray:
    """Select top-k scores with deterministic lower-index tie breaking."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    count = refresh_count(len(values), ratio)
    mask = np.zeros(len(values), dtype=bool)
    if count == 0:
        return mask
    order = np.lexsort((np.arange(len(values)), -values))
    mask[order[:count]] = True
    return mask


@dataclass
class RefreshState:
    """One independent policy/ratio cache state."""

    cache: torch.Tensor
    age: np.ndarray

    @classmethod
    def initialize(cls, current_stage: torch.Tensor, block_count: int) -> "RefreshState":
        return cls(
            cache=current_stage.detach().clone(),
            age=np.zeros(block_count, dtype=np.float32),
        )

    def apply_refresh(
        self,
        current_stage: torch.Tensor,
        refresh_mask: np.ndarray,
        grid: int,
    ) -> None:
        if refresh_mask.shape != self.age.shape:
            raise ValueError("refresh_mask shape does not match block age state")
        self.cache = replace_blocks(
            self.cache,
            current_stage,
            refresh_mask,
            grid=grid,
        )
        self.age += 1.0
        self.age[refresh_mask] = 0.0
