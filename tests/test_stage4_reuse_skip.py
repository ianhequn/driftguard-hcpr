import torch

from driftguard.structured_reuse.pipeline import WholeStage4Cache


def test_reuse_skips_stage4_and_preserves_cached_f4(round5_tiny_model):
    cache = WholeStage4Cache(round5_tiny_model)
    first = torch.randn(1, 32, 8, 8)
    second = torch.randn(1, 32, 8, 8)
    with torch.inference_mode():
        _, state, _ = cache.forward(first, True, None)
        cached = state.cached_f4.clone()
        calls = cache.stage4_forward_calls
        reused, state, refreshed = cache.forward(second, False, state)
    assert not refreshed
    assert cache.stage4_forward_calls == calls
    assert state.age == 1
    assert torch.equal(reused, cached)
    assert torch.equal(state.cached_f4, cached)
