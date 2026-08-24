# DriftGuard Round 2.5 Progress

## Objective

Determine the strongest practical refresh selector under one corrected
sequential simulator, without adding models, features, data, or cache cuts.

## Completed

- Audited cache initialization, block-age updates, refresh timing, decoder
  evaluation, budget rounding, and ranking ties.
- Corrected the task-damage Oracle and its first-frame cache initialization.
- Added and passed 8 refresh-semantics tests.
- Restored the frozen 737-parameter task-risk MLP once under the Round 2
  protocol.
- Ran 3 held-out videos, 50 Random seeds, 8 policies, and 6 budgets.
- Re-ran all deterministic policies and same-cache block diagnostics.
- Profiled input preparation, model forward, ranking, and total selector cost.
- Generated per-video, motion-conditioned, bootstrap, ranking, and age
  diagnostics.
- Generated all required tables and figures.

## Final Status

- Best learned candidate: `CNN Patch+Age`.
- Final selector outcome: `Marginal / No Clear Winner`.
- Scientific conclusion: `MARGINAL / MORE SCIENTIFIC WORK REQUIRED`.
- Real Partial Computation: blocked for now.

Round 2.5 is complete. No hybrid or next-phase implementation was executed.
