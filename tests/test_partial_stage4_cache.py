import torch

from driftguard.partial.stage4 import PartialStage4Engine


def test_selected_cache_updates_and_unselected_cache_stays(round5_tiny_model):
    engine = PartialStage4Engine(round5_tiny_model)
    first = torch.randn(1, 32, 8, 8)
    second = first + torch.randn_like(first) * 0.2
    with torch.inference_mode():
        _, cache, _ = engine.initialize(first)
        before_output = cache.output_tokens.clone()
        before_k = [value.clone() for value in cache.layer_k]
        mask = torch.zeros(16, dtype=torch.bool)
        mask[5] = True
        _, cache, stats = engine.partial_stage4(second, cache, mask)

    selected, _ = engine._token_indices(
        mask,
        cache.height,
        cache.width,
        cache.output_tokens.device,
    )
    all_indices = torch.arange(cache.height * cache.width)
    unselected = all_indices[~torch.isin(all_indices, selected.cpu())]
    assert stats.logical_refresh_blocks == 1
    assert not torch.equal(
        cache.output_tokens.index_select(1, selected),
        before_output.index_select(1, selected),
    )
    assert torch.equal(
        cache.output_tokens.index_select(1, unselected),
        before_output.index_select(1, unselected),
    )
    for current, previous in zip(cache.layer_k, before_k):
        assert torch.equal(
            current.index_select(2, unselected),
            previous.index_select(2, unselected),
        )
