# Algorithm code map

This page connects the algorithm idea to executable code and tests.

| Concept | Implementation | Evidence / tests |
|---|---|---|
| SegFormer stage decomposition | `driftguard/models/segformer.py` | stage shape and loading audits |
| Fixed boundary reuse | `driftguard/structured_reuse/boundaries.py` | `tests/test_hcpr_pipeline.py` |
| HCPR state machine | `driftguard/structured_reuse/hcpr.py` | forced-Full, physical skip, age guard, determinism tests |
| Hardware cost variants | `driftguard/structured_reuse/r11b.py` | `tests/test_round11b_variants.py` |
| Stage-age / visual-risk baselines | `driftguard/structured_reuse/policies.py` | Round 2–6 experiment artifacts |
| Unlabeled drift calibration | `scripts/run_round11a_calibration.py` | `reports/round11a_hcpr_state.json` |
| Fixed quality/latency matrix | `scripts/run_round10_quality.py`, `scripts/run_round10_latency.py` | `results/R10_E03_quality`, `results/R10_E04_latency` |
| HCPR transfer diagnostics | `scripts/run_round11b_quality.py`, `scripts/run_round11b_latency.py` | `results/R11B_E03_transfer`, `results/R11B_E04_latency` |
| Protocol/Pareto audit | `scripts/audit_mmm_pareto_consistency.py` | `reports/post_mmm/pareto_consistency_audit.*` |
| Checkpoint recovery audit | `scripts/audit_checkpoint_inventory.py` | `reports/post_mmm/checkpoint_inventory.*` |

## Core execution semantics

For every video, the first frame executes `Full` and initializes cached F2/F3/F4
features plus pooled boundary references. Later frames always execute the fresh
prefix F1. The controller compares each boundary score with the calibrated
lambda. If a route reuses a suffix, those stages are absent from the current
frame's execution path; the decoder consumes the cached feature directly.

The cache state tracks age independently for the three suffixes. The deepest
stale age propagates its guard upward, so a stale F4 cannot be hidden behind a
shallow route decision. Video boundaries reset the state.
