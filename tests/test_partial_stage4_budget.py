import pytest
import torch

from driftguard.partial.stage4 import PartialStage4Engine


@pytest.mark.parametrize("blocks", [2, 4, 6, 8, 12, 16])
def test_partial_stage4_budget_is_exact(round5_tiny_model, blocks):
    engine = PartialStage4Engine(round5_tiny_model)
    feature = torch.randn(1, 32, 8, 8)
    mask = torch.zeros(16, dtype=torch.bool)
    mask[:blocks] = True
    with torch.inference_mode():
        _, cache, _ = engine.initialize(feature)
        _, _, stats = engine.partial_stage4(feature + 0.1, cache, mask)
    expected_tokens = blocks
    assert stats.logical_refresh_blocks == blocks
    assert stats.logical_tokens == expected_tokens
    assert stats.physical_tokens == expected_tokens
    assert stats.full_tokens == 16
    assert stats.physical_compute_ratio == pytest.approx(blocks / 16)
