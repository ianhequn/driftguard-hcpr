import torch

from driftguard.structured_reuse.pipeline import WholeStage4Cache


def test_whole_cache_first_frame_refreshes_and_allocates_only_f4(
    round5_tiny_model,
):
    cache = WholeStage4Cache(round5_tiny_model)
    f3 = torch.randn(1, 32, 8, 8)
    with torch.inference_mode():
        f4, state, refreshed = cache.forward(f3, False, None)
    assert refreshed
    assert state.initialized
    assert state.age == 0
    assert state.cached_f4 is not None
    assert torch.equal(f4, state.cached_f4)
    assert cache.stage4_forward_calls == 1
    assert state.persistent_bytes == f4.numel() * f4.element_size()
