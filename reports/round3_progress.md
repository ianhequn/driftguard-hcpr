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
