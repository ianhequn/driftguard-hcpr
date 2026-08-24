"""Age-aware visual refresh schedulers for DriftGuard Round 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from driftguard.analysis.refresh_semantics import refresh_count


VisualScoreFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class SchedulerDecision:
    mask: np.ndarray
    cnn_indices: np.ndarray
    tie_group_size: int
    forced_blocks: int
    visual_selected_blocks: int
    forced_overflow: bool
    max_age: int | None

    @property
    def cnn_evaluated_blocks(self) -> int:
        return int(len(self.cnn_indices))

    @property
    def cnn_calls(self) -> int:
        return int(len(self.cnn_indices) > 0)


def _validate_ages(ages: np.ndarray) -> np.ndarray:
    values = np.asarray(ages, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("ages must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("ages must be finite and non-negative")
    return values


def _visual_top_k(
    indices: np.ndarray,
    count: int,
    visual_score_fn: VisualScoreFn,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray(indices, dtype=np.int64)
    if count < 0 or count > len(candidates):
        raise ValueError("visual selection count is outside candidate set")
    if count == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if count == len(candidates):
        return candidates.copy(), np.empty(0, dtype=np.int64)
    scores = np.asarray(visual_score_fn(candidates), dtype=np.float64)
    if scores.shape != (len(candidates),):
        raise ValueError("visual_score_fn returned the wrong shape")
    order = np.lexsort((candidates, -scores))
    return candidates[order[:count]], candidates


def budget_derived_max_age(block_count: int, ratio: float) -> int:
    count = refresh_count(block_count, ratio)
    if count == 0:
        raise ValueError("max age is undefined for a zero refresh budget")
    horizon = int(np.ceil(block_count / count))
    return horizon - 1


def select_afvt(
    ages: np.ndarray,
    ratio: float,
    visual_score_fn: VisualScoreFn,
) -> SchedulerDecision:
    """Age hierarchy first, visual risk only within the K-boundary tie."""

    values = _validate_ages(ages)
    count = refresh_count(len(values), ratio)
    mask = np.zeros(len(values), dtype=bool)
    if count == 0:
        return SchedulerDecision(
            mask=mask,
            cnn_indices=np.empty(0, dtype=np.int64),
            tie_group_size=0,
            forced_blocks=0,
            visual_selected_blocks=0,
            forced_overflow=False,
            max_age=None,
        )

    threshold = np.sort(values)[::-1][count - 1]
    older = np.flatnonzero(values > threshold)
    tie = np.flatnonzero(values == threshold)
    remaining = count - len(older)
    selected_tie, cnn_indices = _visual_top_k(
        tie,
        remaining,
        visual_score_fn,
    )
    mask[older] = True
    mask[selected_tie] = True
    return SchedulerDecision(
        mask=mask,
        cnn_indices=cnn_indices,
        tie_group_size=int(len(tie)),
        forced_blocks=int(len(older)),
        visual_selected_blocks=int(len(selected_tie)),
        forced_overflow=False,
        max_age=None,
    )


def select_magvr(
    ages: np.ndarray,
    ratio: float,
    visual_score_fn: VisualScoreFn,
) -> SchedulerDecision:
    """Budget-derived max-age guard plus visual allocation of free budget."""

    values = _validate_ages(ages)
    count = refresh_count(len(values), ratio)
    mask = np.zeros(len(values), dtype=bool)
    if count == 0:
        return SchedulerDecision(
            mask=mask,
            cnn_indices=np.empty(0, dtype=np.int64),
            tie_group_size=0,
            forced_blocks=0,
            visual_selected_blocks=0,
            forced_overflow=False,
            max_age=None,
        )

    max_age = budget_derived_max_age(len(values), ratio)
    forced = np.flatnonzero(values >= max_age)
    if len(forced) > count:
        forced_ages = values[forced]
        threshold = np.sort(forced_ages)[::-1][count - 1]
        older = forced[forced_ages > threshold]
        tie = forced[forced_ages == threshold]
        remaining = count - len(older)
        selected_tie, cnn_indices = _visual_top_k(
            tie,
            remaining,
            visual_score_fn,
        )
        mask[older] = True
        mask[selected_tie] = True
        return SchedulerDecision(
            mask=mask,
            cnn_indices=cnn_indices,
            tie_group_size=int(len(tie)),
            forced_blocks=int(len(forced)),
            visual_selected_blocks=int(len(selected_tie)),
            forced_overflow=True,
            max_age=max_age,
        )

    mask[forced] = True
    remaining = count - len(forced)
    candidates = np.flatnonzero(values < max_age)
    selected_visual, cnn_indices = _visual_top_k(
        candidates,
        remaining,
        visual_score_fn,
    )
    mask[selected_visual] = True
    return SchedulerDecision(
        mask=mask,
        cnn_indices=cnn_indices,
        tie_group_size=0,
        forced_blocks=int(len(forced)),
        visual_selected_blocks=int(len(selected_visual)),
        forced_overflow=False,
        max_age=max_age,
    )
