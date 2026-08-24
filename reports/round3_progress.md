# DriftGuard Round 3 Progress

## 2026-08-24 - Source-of-Truth Audit

- Phase: Round 2.5 handoff.
- Config: frozen SegFormer-B0, Stage 4, 4x4 blocks, frozen Patch+Age CNN.
- Results: Age remains the practical baseline; Patch+Age is the strongest
  learned candidate; Round 2.5 remains Marginal / No Clear Winner.
- Bugs: none in the frozen Round 2.5 source-of-truth artifacts.
- Decision: evaluate scheduling only.
- Next step: implement and test AFVT and budget-derived MAG-VR.

## 2026-08-24 - Scheduler Semantics

- Phase: implementation started.
- Config: AFVT and MAG-VR have zero tunable hyperparameters.
- Results: pending unit tests.
- Bugs: none recorded.
- Decision: visual inference is candidate-only by scheduler construction.
- Next step: run all previous and Round 3 scheduler tests.

## 2026-08-24 - Regression Tests and Validation Smoke

- Phase: semantics validation and smoke test.
- Config: 1 validation video, 20 frames, budgets 25%, 37.5%, and 50%.
- Results: 19 tests passed; AFVT and MAG-VR respected every budget; cache
  state and age evolution were valid; conditional CNN block counts were
  recorded.
- Bugs: none.
- Decision: no optional Amax validation sweep is justified. Freeze
  `Amax = ceil(16 / K) - 1`.
- Next step: run the frozen configuration once on the 3 held-out test videos.

## 2026-08-24 - Frozen Test Evaluation

- Phase: formal test evaluation.
- Config: 3 frozen test videos; budgets 12.5%, 25%, 37.5%, 50%, 75%, and
  100%; core decisions use 25%, 37.5%, and 50%.
- Results: 5,076 policy-frame rows. AFVT and MAG-VR are no worse than Age at
  all three core budgets. Each has strict negative video-bootstrap confidence
  intervals at 37.5% and 50%.
- Bugs: none. Round 2.5 baselines reproduce to floating-point precision.
- Decision: both new schedulers satisfy the frozen STRONG GO evidence rule.
- Next step: run diagnostics, secondary-metric guards, and profiling.

## 2026-08-24 - Diagnostics and Profiling

- Phase: fairness, visual usage, bootstrap, and selector profiling.
- Config: 1,000 video-level bootstrap resamples; RTX 4090 profiling with
  warmup 100 and 500 measurements.
- Results: AFVT preserves Age max stale runs exactly (`3/2/1`). MAG-VR remains
  controlled at `5/3/1`. At 37.5%, selector latency is 38.404 ms for AFVT and
  39.582 ms for MAG-VR.
- Bugs: none. Stage and total timings are independently measured; high
  variance is retained rather than normalized away.
- Decision: AFVT has the best average core-budget damage and stronger fairness.
- Next step: produce the final scientific and deployment decisions.

## 2026-08-24 - Final Decision

- Phase: Round 3 closure.
- Config: no post-test tuning or method changes.
- Results: `SCIENTIFIC RESULT = STRONG GO`; `ALGORITHM WINNER = AFVT`.
  Pixel Disagreement triggers a single-budget warning at 37.5%.
- Bugs: none. All 7 figures passed visual QA; JSON/CSV validation passed; 19
  cloud tests passed; local Python compilation passed.
- Decision: `ALGORITHM LOCKED - PIPELINE OPTIMIZATION REQUIRED`.
- Next step: stop. GPU-resident selector pipeline optimization may be proposed
  as a separate round, but Partial Computation is not authorized here.
