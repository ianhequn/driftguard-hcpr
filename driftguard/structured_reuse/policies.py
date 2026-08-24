"""Deployable frame-level policies for whole-Stage-4 refresh."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def aggregate_frame_risk(
    block_risks: np.ndarray,
    aggregation: str,
    top_k: int = 4,
) -> float:
    values = np.asarray(block_risks, dtype=np.float64)
    if values.shape != (16,):
        raise ValueError("block_risks must contain exactly 16 values")
    if aggregation == "max":
        return float(values.max())
    if aggregation == "mean":
        return float(values.mean())
    if aggregation == "p90":
        return float(np.quantile(values, 0.90))
    if aggregation == "topk_mean":
        if not 1 <= top_k <= len(values):
            raise ValueError("top_k must be between 1 and 16")
        return float(np.partition(values, -top_k)[-top_k:].mean())
    raise ValueError(f"Unsupported aggregation: {aggregation}")


@dataclass(frozen=True)
class FixedIntervalPolicy:
    interval: int

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError("interval must be positive")

    def decide(self, frame_index: int, cache_age: int, **_) -> bool:
        del cache_age
        return frame_index % self.interval == 0


@dataclass(frozen=True)
class StageAgePolicy:
    threshold: int

    def __post_init__(self) -> None:
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")

    def decide(self, frame_index: int, cache_age: int, **_) -> bool:
        return frame_index == 0 or cache_age >= self.threshold


@dataclass(frozen=True)
class VisualRiskPolicy:
    threshold: float

    def decide(
        self,
        frame_index: int,
        cache_age: int,
        *,
        frame_risk: float | None = None,
        **_,
    ) -> bool:
        del cache_age
        if frame_index == 0:
            return True
        if frame_risk is None:
            raise ValueError("VisualRiskPolicy requires current frame_risk")
        return frame_risk > self.threshold


@dataclass(frozen=True)
class StageAgeVisualPolicy:
    max_age: int
    threshold: float

    def __post_init__(self) -> None:
        if self.max_age < 0:
            raise ValueError("max_age must be non-negative")

    def decide(
        self,
        frame_index: int,
        cache_age: int,
        *,
        frame_risk: float | None = None,
        **_,
    ) -> bool:
        if frame_index == 0 or cache_age >= self.max_age:
            return True
        if frame_risk is None:
            raise ValueError("StageAgeVisualPolicy requires current frame_risk")
        return frame_risk > self.threshold
