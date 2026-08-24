import numpy as np
import torch

from driftguard.analysis.age_aware_schedulers import (
    budget_derived_max_age,
    select_afvt,
    select_magvr,
)
from driftguard.analysis.refresh_semantics import RefreshState, select_top_k


def _scores(values):
    values = np.asarray(values, dtype=np.float64)

    def score(indices):
        return values[indices]

    return score


def _stage(offset=0):
    return torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) + offset


def test_budget_derived_max_age():
    assert budget_derived_max_age(16, 0.25) == 3
    assert budget_derived_max_age(16, 0.375) == 2
    assert budget_derived_max_age(16, 0.5) == 1


def test_afvt_distinct_ages_matches_age():
    ages = np.arange(16, dtype=np.float32)
    decision = select_afvt(ages, 0.25, _scores(np.arange(16)[::-1]))
    assert np.array_equal(decision.mask, select_top_k(ages, 0.25))
    assert decision.cnn_evaluated_blocks == 0


def test_afvt_equal_ages_matches_visual_top_k():
    ages = np.zeros(16, dtype=np.float32)
    visual = np.linspace(0.0, 1.0, 16)
    decision = select_afvt(ages, 0.25, _scores(visual))
    assert np.array_equal(decision.mask, select_top_k(visual, 0.25))
    assert decision.tie_group_size == 16
    assert decision.cnn_evaluated_blocks == 16


def test_afvt_never_promotes_younger_block_over_older_block():
    ages = np.asarray([3, 3, 2, 2] + [0] * 12, dtype=np.float32)
    visual = np.asarray([0, 0, 1, 1] + [100] * 12, dtype=np.float32)
    decision = select_afvt(ages, 0.25, _scores(visual))
    assert decision.mask[:4].all()
    assert not decision.mask[4:].any()


def test_magvr_refreshes_all_forced_blocks_when_budget_allows():
    ages = np.asarray([3, 3, 3, 3] + [0] * 12, dtype=np.float32)
    decision = select_magvr(ages, 0.25, _scores(np.arange(16)))
    assert decision.mask[:4].all()
    assert decision.forced_blocks == 4
    assert not decision.forced_overflow


def test_magvr_forced_overflow_respects_budget():
    ages = np.asarray([4] * 3 + [3] * 5 + [0] * 8, dtype=np.float32)
    visual = np.arange(16, dtype=np.float32)
    decision = select_magvr(ages, 0.25, _scores(visual))
    assert decision.mask.sum() == 4
    assert decision.mask[:3].all()
    assert decision.mask[7]
    assert decision.forced_overflow


def test_full_refresh_matches_current_for_both_schedulers():
    current = _stage()
    next_stage = _stage(100)
    for scheduler in (select_afvt, select_magvr):
        state = RefreshState.initialize(current, block_count=16)
        decision = scheduler(
            state.age,
            1.0,
            _scores(np.arange(16, dtype=np.float32)),
        )
        state.apply_refresh(next_stage, decision.mask, grid=4)
        assert torch.equal(state.cache, next_stage)
        assert np.array_equal(state.age, np.zeros(16, dtype=np.float32))


def test_cache_age_reset_and_increment():
    state = RefreshState.initialize(_stage(), block_count=16)
    visual = np.arange(16, dtype=np.float32)
    decision = select_afvt(state.age, 0.25, _scores(visual))
    state.apply_refresh(_stage(100), decision.mask, grid=4)
    assert np.array_equal(state.age[decision.mask], np.zeros(4))
    assert np.array_equal(state.age[~decision.mask], np.ones(12))


def test_policies_keep_independent_cache_state():
    current = _stage()
    afvt_state = RefreshState.initialize(current, block_count=16)
    magvr_state = RefreshState.initialize(current, block_count=16)
    visual = np.arange(16, dtype=np.float32)
    afvt = select_afvt(afvt_state.age, 0.25, _scores(visual))
    magvr = select_magvr(magvr_state.age, 0.25, _scores(visual[::-1]))
    afvt_state.apply_refresh(_stage(100), afvt.mask, grid=4)
    magvr_state.apply_refresh(_stage(200), magvr.mask, grid=4)
    assert afvt_state.cache.data_ptr() != magvr_state.cache.data_ptr()
    assert not torch.equal(afvt_state.cache, magvr_state.cache)
