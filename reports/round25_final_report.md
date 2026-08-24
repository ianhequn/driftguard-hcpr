# DriftGuard Round 2.5 Final Report

## Decision

**FINAL SELECTOR = Marginal / No Clear Winner**

**SCIENTIFIC CONCLUSION = MARGINAL / MORE SCIENTIFIC WORK REQUIRED**

`CNN Patch+Age` is the strongest learned candidate, but it does not provide a
stable and practical enough advantage over `Age` to lock the deployment
selector. Do not begin Real Partial Computation from this result.

## A. Simulator Audit

The Round 2 deployable cache semantics were correct:

- Independent cache and block-age state for every policy and budget.
- First-frame cache initialization and age 0 initialization.
- Pre-refresh scoring, selected-block replacement, then decoder evaluation.
- Refreshed age reset to 0; unrefreshed age incremented by 1.
- Budget rule `ceil(16 * ratio)`.
- Age ranks independent block ages from high to low.

Corrections made in Round 2.5:

1. The historical Round 2 Oracle used feature drift rather than true task
   damage.
2. The initial Round 2.5 Oracle cache was initialized one evaluated frame
   later than deployable policies.
3. Ranking ties now use deterministic lower-block-index ordering.

The corrected deterministic rerun reproduced deployable results to numerical
precision. The Oracle correction changed only the upper bound. All 8 simulator
tests passed.

## B. Main Refresh Results

Metric: mean test `LogitRelativeL2`, lower is better.

| Method | Params | Selector ms | 25% | 37.5% | 50% |
|---|---:|---:|---:|---:|---:|
| Random | 0 | - | 0.022778 | 0.017726 | 0.014126 |
| Residual Mean | 0 | 9.660 | 0.029987 | 0.023195 | 0.017216 |
| Residual P90 | 0 | 43.769 | 0.030456 | 0.023415 | 0.015388 |
| Age | 0 | 0.026 | 0.018285 | 0.014414 | 0.010677 |
| Old Tiny MLP | 737 | 11.246 | 0.021630 | 0.017765 | 0.014402 |
| CNN All | 5,865 | 134.991 | 0.020173 | 0.015808 | 0.012845 |
| CNN Patch+Age | 5,737 | 96.769 | **0.017749** | **0.013717** | 0.010679 |
| Corrected Oracle | - | - | 0.015237 | 0.011314 | 0.008743 |

Relative to Age, Patch+Age changes task damage by:

- 25%: `-2.93%`
- 37.5%: `-4.83%`
- 50%: `+0.02%`

It wins two core budgets, but the 50% result is effectively tied and slightly
worse.

## C. Per-Video Results

Patch+Age versus Age:

| Ratio | Video | Motion | Age | Patch+Age | Winner |
|---:|---|---|---:|---:|---|
| 25% | asl_brother | high | 0.020990 | 0.021112 | Age |
| 25% | asl_milk | low | 0.015255 | 0.014055 | Patch+Age |
| 25% | asl_school | medium | 0.018610 | 0.018080 | Patch+Age |
| 37.5% | asl_brother | high | 0.016761 | 0.016199 | Patch+Age |
| 37.5% | asl_milk | low | 0.011639 | 0.010701 | Patch+Age |
| 37.5% | asl_school | medium | 0.014841 | 0.014252 | Patch+Age |
| 50% | asl_brother | high | 0.012367 | 0.012090 | Patch+Age |
| 50% | asl_milk | low | 0.008484 | 0.008674 | Age |
| 50% | asl_school | medium | 0.011179 | 0.011274 | Age |

Patch+Age wins 6 of 9 video-budget comparisons. Its strongest stability occurs
at 37.5%, where it wins low-, medium-, and high-motion videos.

Bootstrap at the video level:

| Ratio | Patch+Age minus Age | 95% CI | Better videos |
|---:|---:|---:|---:|
| 25% | -0.000536 | [-0.001200, 0.000122] | 2/3 |
| 37.5% | -0.000696 | [-0.000938, -0.000562] | 3/3 |
| 50% | +0.000003 | [-0.000277, 0.000190] | 1/3 |

Only the 37.5% confidence interval is strictly below zero.

## D. Selector Quality

Static predictor metrics use the frozen Round 2 task-risk dataset. Oracle
overlap uses each policy's own pre-refresh sequential cache state.

| Method | AUPRC | Spearman | Top-4 | Top-6 | Top-8 |
|---|---:|---:|---:|---:|---:|
| Residual Mean | 0.7672 | 0.6956 | 0.202 | 0.286 | 0.403 |
| Residual P90 | 0.7451 | 0.6551 | 0.197 | 0.295 | 0.416 |
| Age | 0.3109 | 0.6132 | 0.391 | 0.521 | **0.667** |
| Old Tiny MLP | 0.8150 | 0.7561 | 0.240 | 0.348 | 0.476 |
| CNN All | 0.8003 | 0.7768 | 0.319 | 0.420 | 0.546 |
| CNN Patch+Age | **0.8159** | 0.7367 | **0.442** | **0.544** | 0.653 |

Patch+Age adds useful ranking information beyond age at Top-4 and Top-6.
However, higher local Oracle overlap does not guarantee better long-horizon
cache evolution at every refresh budget.

## E. Selector Overhead

RTX 4090, warmup 100 and measurement 500 for end-to-end paths. Model-only
CUDA forward was rechecked with warmup 200 and measurement 1,000 after GPU
clock priming.

| Method | Input prep ms | Model ms | Ranking ms | Total ms | P95 total ms |
|---|---:|---:|---:|---:|---:|
| Residual Mean | 11.586 | 0 | 0.025 | 9.660 | 9.852 |
| Residual P90 | 41.186 | 0 | 0.026 | 43.769 | 98.067 |
| Age | 0.019 | 0 | 0.025 | 0.026 | 0.028 |
| Old Tiny MLP | 11.848 | 0.244 | 0.053 | 11.246 | 12.327 |
| CNN All | 123.571 | 0.555 | 0.157 | 134.991 | 196.818 |
| CNN Patch+Age | 86.375 | 0.499 | 0.154 | 96.769 | 105.785 |

Stage means were measured in separate runs and are diagnostic components; they
are not expected to sum exactly to the independently measured total.

The CNN forward itself remains small. The current practical cost is dominated
by Python/CPU residual-patch and metadata preparation. Patch+Age costs roughly
3,710 times the Age selector in the measured implementation.

## F. Why Age Is Strong

Mean same-cache single-block task damage rises monotonically with age:

| Cache age | Mean task damage |
|---:|---:|
| 0 | 0.003383 |
| 1 | 0.004636 |
| 2 | 0.005857 |
| 3 | 0.006285 |

Spearman correlation between age and task damage remains positive in every
motion group:

- High motion: `0.392`
- Low motion: `0.481`
- Medium motion: `0.316`

At the 37.5% representative point, Oracle Top-6 blocks have age distribution:

- Age 0: 17.0%
- Age 1: 38.5%
- Age 2: 44.4%

Age is strong because stale lifetime is a real task-damage prior and because
its round-robin-like refresh behavior prevents starvation across blocks. This
is a sequential cache-management advantage, not merely a static classifier
advantage.

## G. Learned Model Diagnostic

Old Tiny MLP and CNN All do not beat Age at any core budget or any of the nine
video-budget comparisons. Their strong static task-risk classification metrics
do not translate into stronger sequential cache control.

Patch+Age does contain information beyond age:

- Better Top-4 and Top-6 same-cache Oracle overlap.
- Lower main damage at 25% and 37.5%.
- Wins all motion groups at 37.5%.

But the extra visual signal is not consistently useful:

- It loses two of three videos at 50%.
- Its 50% pooled result is slightly worse than Age.
- At 37.5%, its pixel disagreement is `0.010670`, worse than Age's `0.008350`,
  despite better primary LogitRelativeL2.
- Its total selector cost is about `96.8 ms/frame`.

Therefore, the learned visual correction is scientifically promising but not
yet a deployable winner.

## H. Final Winner

**Marginal / No Clear Winner**

Best learned candidate: **TinyRiskCNN Patch+Age**.

Strongest practical baseline: **Age**.

Neither statement alone is enough to lock the final algorithm.

## I. Scientific Conclusion

**MARGINAL / MORE SCIENTIFIC WORK REQUIRED**

There is not sufficient evidence to lock a learned selector and begin Real
Partial Computation. Declaring Patch+Age the winner would ignore its
budget/video instability, secondary-metric inconsistency, and large current
selector overhead. Declaring Age fully dominant would ignore Patch+Age's real
and statistically clear 37.5% gain.

The Round 2.5 decision itself is locked. The deployment selector is not.

## J. Next Step

No next-phase implementation is authorized by this round. Real Partial
Computation remains blocked.

The evidence supports two future hypotheses only:

1. `Age Prior + Visual Risk Correction`
2. `Age-Aware Cache Lifetime`

Neither hypothesis was implemented in Round 2.5.
