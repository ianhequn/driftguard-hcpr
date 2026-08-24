import torch

from driftguard.partial.stage4 import PartialStage4Engine


def test_full_refresh_matches_full_stage4(round5_tiny_model):
    engine = PartialStage4Engine(round5_tiny_model)
    first = torch.randn(1, 32, 8, 8)
    second = torch.randn(1, 32, 8, 8)
    with torch.inference_mode():
        _, cache, _ = engine.initialize(first)
        expected, _ = engine._full_stage4_with_cache(second)
        actual, _, stats = engine.partial_stage4(
            second,
            cache,
            torch.ones(16, dtype=torch.bool),
        )
    assert stats.full_refresh
    assert stats.physical_compute_ratio == 1.0
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)
