# Round 2.5 Refresh Simulator Audit

## Verdict

The deployable Round 2 cache semantics were correct:

- The first decoded frame initialized each policy and refresh-budget cache.
- Every policy and refresh ratio owned independent cache and block-age state.
- Scores were computed from pre-refresh state.
- Selected Stage 4 blocks were refreshed before decoder task evaluation.
- Refreshed block age reset to 0; unrefreshed block age increased by 1.
- The budget was `ceil(16 * ratio)`.
- Age policy ranked independent block ages from highest to lowest.

Age did not use global age, future information, or a different reset rule.

## Corrections

1. The historical Round 2 Oracle ranked feature drift. It was not a true
   task-damage Oracle. Round 2.5 replaces it with single-block
   `LogitRelativeL2` task damage.
2. The first Round 2.5 closure implementation initialized Oracle state one
   evaluated frame later than deployable policies. This affected only the
   Oracle upper bound. Oracle now initializes on the same first frame.
3. Ranking ties now use deterministic lower-block-index ordering.

## Validation

- Cloud test environment: RTX 4090 project environment.
- Tests: 8 passed.
- Corrected deterministic rerun: 5,922 frame-policy rows.
- Same-cache selector diagnostics: 40,608 block rows.
- Age, MLP, CNN, and Residual Mean results reproduced to numerical precision.
- Residual P90 changed by at most approximately `6.9e-06` at a core budget.
- Corrected Oracle is worse than the faulty Oracle, as expected, while
  deployable method ordering is unchanged.

## Scope

Oracle remains analysis-only. No Oracle signal is used by a deployable
selector, checkpoint, training label selection, or policy update.
