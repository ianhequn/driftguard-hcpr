import torch

from driftguard.structured_reuse.boundaries import BoundaryCacheState, StructuredBoundaryReusePipeline
from driftguard.structured_reuse.hcpr import HardwareCalibratedProgressiveReusePipeline


def test_hcpr_force_full_matches_native_segformer(round5_tiny_model):
    pipeline = HardwareCalibratedProgressiveReusePipeline(round5_tiny_model)
    pixels = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        expected = round5_tiny_model(pixel_values=pixels, return_dict=True).logits
        actual, state, stats = pipeline.forward(pixels, forced_route="full")
    assert torch.equal(actual, expected)
    assert state.initialized
    assert stats["route"] == "Full"
    assert stats["stage_calls"] == {
        "stage1": 1, "stage2": 1, "stage3": 1, "stage4": 1, "decoder": 1
    }


def test_hcpr_progressive_routes_physically_skip_deeper_stages(round5_tiny_model):
    pipeline = HardwareCalibratedProgressiveReusePipeline(
        round5_tiny_model, lambda_value=1e6, max_stale_run=None
    )
    pixels = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        _, state, _ = pipeline.forward(pixels, state=None)
        _, state, stats = pipeline.forward(pixels, state=state)
    assert stats["route"] == "S2-S4"
    assert stats["stage_calls"] == {
        "stage1": 1, "stage2": 0, "stage3": 0, "stage4": 0, "decoder": 1
    }
    assert state.age2 == state.age3 == state.age4 == 1


def test_hcpr_forced_routes_match_round10_boundary_semantics(round5_tiny_model):
    pixels = torch.randn(1, 3, 64, 64)
    for boundary, forced in (("s4", "s4"), ("s3_s4", "s3_s4"), ("s2_s4", "s2_s4")):
        hcpr = HardwareCalibratedProgressiveReusePipeline(round5_tiny_model)
        reference = StructuredBoundaryReusePipeline(round5_tiny_model, boundary)
        hcpr_state = None
        reference_state = BoundaryCacheState()
        with torch.inference_mode():
            first_h, hcpr_state, _ = hcpr.forward(pixels, state=hcpr_state, forced_route=forced)
            first_r, reference_state, _ = reference.forward(pixels, refresh=True, state=reference_state)
            second_h, hcpr_state, _ = hcpr.forward(pixels + 0.01, state=hcpr_state, forced_route=forced)
            second_r, reference_state, _ = reference.forward(pixels + 0.01, refresh=False, state=reference_state)
        assert torch.equal(first_h, first_r)
        assert torch.equal(second_h, second_r)


def test_hcpr_age_guard_refreshes_at_bound(round5_tiny_model):
    pipeline = HardwareCalibratedProgressiveReusePipeline(
        round5_tiny_model, lambda_value=1e6, max_stale_run=2
    )
    pixels = torch.randn(1, 3, 64, 64)
    state = None
    routes = []
    with torch.inference_mode():
        for _ in range(4):
            _, state, stats = pipeline.forward(pixels, state=state)
            routes.append(stats["route"])
    assert routes == ["Full", "S2-S4", "S2-S4", "Full"]
    assert state.age2 == state.age3 == state.age4 == 0


def test_hcpr_is_deterministic_for_same_sequence(round5_tiny_model):
    first = HardwareCalibratedProgressiveReusePipeline(
        round5_tiny_model, stage_costs={2: 0.5, 3: 1.0, 4: 2.0},
        drift_scales={2: 0.1, 3: 0.2, 4: 0.3}, lambda_value=0.7,
    )
    second = HardwareCalibratedProgressiveReusePipeline(
        round5_tiny_model, stage_costs={2: 0.5, 3: 1.0, 4: 2.0},
        drift_scales={2: 0.1, 3: 0.2, 4: 0.3}, lambda_value=0.7,
    )
    pixels = [torch.randn(1, 3, 64, 64) for _ in range(8)]
    routes_a, routes_b = [], []
    state_a = state_b = None
    with torch.inference_mode():
        for value in pixels:
            _, state_a, stats_a = first.forward(value, state=state_a)
            _, state_b, stats_b = second.forward(value, state=state_b)
            routes_a.append(stats_a["route"])
            routes_b.append(stats_b["route"])
    assert routes_a == routes_b
