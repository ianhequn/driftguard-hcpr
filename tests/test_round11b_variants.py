import pytest
import torch

from driftguard.structured_reuse.hcpr import HardwareCalibratedProgressiveReusePipeline
from driftguard.structured_reuse.r11b import VARIANTS, make_hcpr_variant


def configs():
    scales = {2: 0.1, 3: 0.2, 4: 0.3}
    old = {"k2": 0.2, "k3": 0.3, "k4": 0.4}
    new = {"k2": 0.5, "k3": 0.6, "k4": 0.7}
    return scales, old, new


def test_variants_change_only_cost_vector(round5_tiny_model):
    scales, old, new = configs()
    new_cost = make_hcpr_variant(round5_tiny_model, "HCPR-NewCost", 1.0, scales, old, new)
    old_cost = make_hcpr_variant(round5_tiny_model, "HCPR-OldCost", 1.0, scales, old, new)
    drift_only = make_hcpr_variant(round5_tiny_model, "ProgressiveDriftOnly", 1.0, scales, old, new)
    assert new_cost.drift_scales == old_cost.drift_scales == drift_only.drift_scales == scales
    assert new_cost.stage_costs == {2: 0.5, 3: 0.6, 4: 0.7}
    assert old_cost.stage_costs == {2: 0.2, 3: 0.3, 4: 0.4}
    assert drift_only.stage_costs == {2: 1.0, 3: 1.0, 4: 1.0}


def test_variant_route_is_deterministic(round5_tiny_model):
    scales, old, new = configs()
    a = make_hcpr_variant(round5_tiny_model, "HCPR-NewCost", 0.8, scales, old, new)
    b = make_hcpr_variant(round5_tiny_model, "HCPR-NewCost", 0.8, scales, old, new)
    pixels = [torch.randn(1, 3, 64, 64) for _ in range(5)]
    routes_a, routes_b = [], []
    state_a = state_b = None
    with torch.inference_mode():
        for pixel in pixels:
            _, state_a, stats_a = a.forward(pixel, state=state_a)
            _, state_b, stats_b = b.forward(pixel, state=state_b)
            routes_a.append(stats_a["route"]); routes_b.append(stats_b["route"])
    assert routes_a == routes_b


def test_variant_keeps_age_guard_and_physical_skip(round5_tiny_model):
    scales, old, new = configs()
    pipeline = make_hcpr_variant(round5_tiny_model, "ProgressiveDriftOnly", 1e6, scales, old, new, max_stale_run=2)
    pixel = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        _, state, _ = pipeline.forward(pixel)
        _, state, stats = pipeline.forward(pixel, state=state)
    assert stats["stage_calls"]["stage1"] == 1
    assert stats["stage_calls"]["stage4"] == 0
    assert state.max_stale_run == 2
