import numpy as np
import torch

from driftguard.analysis.refresh_semantics import (
    RefreshState,
    refresh_count,
    select_top_k,
    validate_analysis_only_policies,
)


def _stage(values):
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 4, 4)


def test_budget_rounding_and_deterministic_ties():
    assert refresh_count(16, 0.0) == 0
    assert refresh_count(16, 0.25) == 4
    assert refresh_count(16, 0.375) == 6
    assert refresh_count(16, 1.0) == 16
    mask = select_top_k(np.ones(16), 0.25)
    assert np.flatnonzero(mask).tolist() == [0, 1, 2, 3]


def test_zero_ratio_selects_no_blocks():
    mask = select_top_k(np.arange(16, dtype=np.float32), 0.0)
    assert not mask.any()


def test_one_ratio_selects_every_block():
    mask = select_top_k(np.arange(16, dtype=np.float32), 1.0)
    assert mask.all()


def test_initial_cache_and_zero_refresh():
    current = _stage(range(16))
    state = RefreshState.initialize(current, block_count=4)
    state.apply_refresh(current + 1, np.zeros(4, dtype=bool), grid=2)
    assert torch.equal(state.cache, current)
    assert state.age.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_full_refresh_matches_current_and_resets_age():
    current = _stage(range(16))
    state = RefreshState.initialize(current, block_count=4)
    mask = np.ones(4, dtype=bool)
    state.apply_refresh(current + 1, mask, grid=2)
    assert torch.equal(state.cache, current + 1)
    assert state.age.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_single_block_refresh_only_updates_that_block():
    current = _stage(range(16))
    next_stage = current + 100
    state = RefreshState.initialize(current, block_count=4)
    mask = np.array([False, True, False, False])
    state.apply_refresh(next_stage, mask, grid=2)
    expected = current.clone()
    expected[:, :, :2, 2:] = next_stage[:, :, :2, 2:]
    assert torch.equal(state.cache, expected)
    assert state.age.tolist() == [1.0, 0.0, 1.0, 1.0]


def test_same_semantics_for_wrapped_scores():
    current = _stage(range(16))
    state_a = RefreshState.initialize(current, block_count=4)
    state_b = RefreshState.initialize(current, block_count=4)
    scores = np.array([0.1, 0.9, 0.4, 0.2])
    mask_a = select_top_k(scores, 0.25)
    mask_b = select_top_k(scores, 0.25)
    state_a.apply_refresh(current + 1, mask_a, grid=2)
    state_b.apply_refresh(current + 1, mask_b, grid=2)
    assert torch.equal(state_a.cache, state_b.cache)
    assert np.array_equal(state_a.age, state_b.age)


def test_oracle_is_analysis_only():
    validate_analysis_only_policies(("age", "cnn_patch_age"))
    try:
        validate_analysis_only_policies(("age", "oracle"))
    except ValueError as error:
        assert "oracle" in str(error)
    else:
        raise AssertionError("oracle must not be accepted as deployable")
