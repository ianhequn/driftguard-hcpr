import torch

from driftguard.structured_reuse.pipeline import WholeStage4Cache


def test_refresh_replaces_cache_and_resets_age(round5_tiny_model):
    cache = WholeStage4Cache(round5_tiny_model)
    first = torch.randn(1, 32, 8, 8)
    second = first + 0.5
    with torch.inference_mode():
        _, state, _ = cache.forward(first, True, None)
        _, state, _ = cache.forward(first, False, state)
        previous = state.cached_f4.clone()
        current, state, refreshed = cache.forward(second, True, state)
    assert refreshed
    assert state.age == 0
    assert cache.stage4_forward_calls == 2
    assert not torch.equal(current, previous)
    assert torch.equal(current, state.cached_f4)
