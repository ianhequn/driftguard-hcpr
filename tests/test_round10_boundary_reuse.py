import pytest

from driftguard.structured_reuse.boundaries import (
    BOUNDARIES,
    BoundaryCacheState,
    StructuredBoundaryReusePipeline,
)


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_full_refresh_executes_all_stages(round5_tiny_model, boundary):
    torch = pytest.importorskip("torch")
    pipeline = StructuredBoundaryReusePipeline(round5_tiny_model, boundary)
    pixels = torch.randn(1, 3, 64, 64)
    logits, state, stats = pipeline.forward(pixels, refresh=True, state=None)

    assert torch.isfinite(logits).all()
    assert state.initialized
    assert stats["actual_refresh"] is True
    assert stats["stage_calls"] == {
        "stage1": 1, "stage2": 1, "stage3": 1, "stage4": 1, "decoder": 1
    }


@pytest.mark.parametrize("boundary,expected", [
    ("s4", {"stage1": 2, "stage2": 2, "stage3": 2, "stage4": 1, "decoder": 2}),
    ("s3_s4", {"stage1": 2, "stage2": 2, "stage3": 1, "stage4": 1, "decoder": 2}),
    ("s2_s4", {"stage1": 2, "stage2": 1, "stage3": 1, "stage4": 1, "decoder": 2}),
])
def test_reuse_skips_the_configured_suffix(round5_tiny_model, boundary, expected):
    torch = pytest.importorskip("torch")
    pipeline = StructuredBoundaryReusePipeline(round5_tiny_model, boundary)
    pixels = torch.randn(1, 3, 64, 64)
    _, state, _ = pipeline.forward(pixels, refresh=True, state=BoundaryCacheState())
    logits, state, stats = pipeline.forward(pixels + 0.01, refresh=False, state=state)

    assert torch.isfinite(logits).all()
    assert stats["actual_refresh"] is False
    assert stats["stage_calls"] == expected
    assert state.age == 1


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_cache_is_replaced_on_refresh(round5_tiny_model, boundary):
    torch = pytest.importorskip("torch")
    pipeline = StructuredBoundaryReusePipeline(round5_tiny_model, boundary)
    pixels = torch.randn(1, 3, 64, 64)
    _, state, _ = pipeline.forward(pixels, refresh=True)
    old = tuple(t.clone() for t in (state.cached_f2, state.cached_f3, state.cached_f4))
    _, state, _ = pipeline.forward(pixels + 0.1, refresh=True, state=state)

    assert state.age == 0
    assert any(not torch.equal(before, after) for before, after in zip(old, (state.cached_f2, state.cached_f3, state.cached_f4)))
