"""Round 11B configuration helpers for frozen HCPR transfer variants."""

from __future__ import annotations

from typing import Any

from .hcpr import HardwareCalibratedProgressiveReusePipeline


VARIANTS = ("HCPR-NewCost", "HCPR-OldCost", "ProgressiveDriftOnly")


def _stage_costs(values: dict[str, Any]) -> dict[int, float]:
    return {stage: float(values[f"k{stage}"]) for stage in (2, 3, 4)}


def make_hcpr_variant(
    model: Any,
    variant: str,
    lambda_value: float,
    drift_scales: dict[int | str, float],
    old_costs: dict[str, Any],
    new_costs: dict[str, Any],
    max_stale_run: int | None = 7,
) -> HardwareCalibratedProgressiveReusePipeline:
    """Build one variant without changing HCPR execution semantics."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if variant == "HCPR-NewCost":
        costs = _stage_costs(new_costs)
    elif variant == "HCPR-OldCost":
        costs = _stage_costs(old_costs)
    else:
        costs = {2: 1.0, 3: 1.0, 4: 1.0}
    return HardwareCalibratedProgressiveReusePipeline(
        model,
        stage_costs=costs,
        drift_scales=drift_scales,
        lambda_value=lambda_value,
        max_stale_run=max_stale_run,
    )
