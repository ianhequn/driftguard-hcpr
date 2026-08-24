# DriftGuard Round 3 Final Report

## Decision

```text
SCIENTIFIC RESULT: STRONG GO
ALGORITHM WINNER: AFVT
DEPLOYMENT READINESS:
ALGORITHM LOCKED - PIPELINE OPTIMIZATION REQUIRED
```

Round 3 answers the frozen research question positively: temporal fairness
plus local visual risk consistently improves the Age scheduler on the primary
metric. AFVT is the algorithm winner because it has the lowest average damage
across the three core budgets and exactly preserves Age's starvation control.

This is an algorithm lock, not deployment approval. At the representative
37.5% budget, AFVT costs `38.404 ms/frame`, far above the project's internal
`2 ms/frame` readiness line. Real Partial Computation remains blocked.

## A. Experimental Integrity

- Backbone: SegFormer-B0.
- Cache cut: Stage 4 / Cut 4.
- Spatial cache: 4x4, 16 blocks per frame.
- Predictor: frozen TinyRiskCNN Patch+Age, 5,737 parameters.
- Test videos: the frozen low-, medium-, and high-motion videos from Round 2.
- Core budgets: 25%, 37.5%, and 50%.
- MAG-VR rule: `Amax = ceil(16 / K) - 1`.
- Validation: one video, 20 frames, core budgets only.
- Optional Amax sweep: not used.
- Formal test: one run after configuration freeze.
- Bootstrap: 1,000 video-level resamples.
- Formal policy-frame records: 5,076.
- Regression result: 19 tests passed.
- Round 3 baseline reruns reproduce Round 2.5 to floating-point precision.

No predictor retraining, test-set tuning, split change, new data, weighted
Age-risk score, Partial Computation, or deployment optimization was used.

## B. Main Result

Metric: mean test `LogitRelativeL2`, lower is better.

| Method | 25% | 37.5% | 50% | Selector ms at 37.5% | P95 ms |
|---|---:|---:|---:|---:|---:|
| Random | 0.022778 | 0.017726 | 0.014126 | - | - |
| Residual | 0.029987 | 0.023195 | 0.017216 | 9.660 | 9.852 |
| Age | 0.018285 | 0.014414 | 0.010677 | 0.059 | 0.069 |
| Pure CNN | 0.017749 | 0.013717 | 0.010679 | 52.799 | 98.791 |
| AFVT | **0.017959** | 0.012979 | **0.010579** | **38.404** | 94.550 |
| MAG-VR | 0.018141 | **0.012927** | **0.010579** | 39.582 | 96.471 |
| Oracle | 0.015237 | 0.011314 | 0.008743 | - | - |

Both new methods are no worse than Age at all three core budgets. MAG-VR is
slightly better than AFVT at 37.5%, and they tie at 50%, but AFVT is better at
25% and has lower mean damage across the three core budgets.

## C. Statistical Evidence

AFVT minus Age:

| Budget | Mean delta | 95% CI | Better videos |
|---:|---:|---:|---:|
| 25% | -0.000326 | [-0.000623, +0.000049] | 2/3 |
| 37.5% | -0.001435 | [-0.002172, -0.000889] | 3/3 |
| 50% | -0.0000976 | [-0.000194, -0.0000275] | 3/3 |

MAG-VR minus Age:

| Budget | Mean delta | 95% CI | Better videos |
|---:|---:|---:|---:|
| 25% | -0.000144 | [-0.000426, +0.000267] | 2/3 |
| 37.5% | -0.001487 | [-0.002244, -0.000942] | 3/3 |
| 50% | -0.0000976 | [-0.000194, -0.0000275] | 3/3 |

Each new method has two strictly negative confidence intervals and two core
budgets with 3/3 video wins. This satisfies the frozen STRONG GO rule.

## D. Secondary Metrics

| Method | Budget | KL | Pixel Disagreement |
|---|---:|---:|---:|
| Age | 25% | 0.004740 | 0.011668 |
| AFVT | 25% | 0.004282 | 0.011275 |
| MAG-VR | 25% | 0.004351 | 0.011431 |
| Age | 37.5% | 0.002860 | 0.008350 |
| AFVT | 37.5% | 0.002393 | 0.008814 |
| MAG-VR | 37.5% | 0.002369 | 0.009069 |
| Age | 50% | 0.001600 | 0.006087 |
| AFVT | 50% | 0.001491 | 0.005988 |
| MAG-VR | 50% | 0.001491 | 0.005988 |

`SECONDARY METRIC WARNING` applies at 37.5%:

- AFVT Pixel Disagreement is about 5.6% worse than Age.
- MAG-VR Pixel Disagreement is about 8.6% worse than Age.

The warning occurs at one core budget, not multiple budgets. Therefore it must
remain visible but does not block STRONG GO under the frozen project rule. KL
supports the primary result at every core budget.

## E. Fairness and Starvation

Global maximum stale run at 25% / 37.5% / 50%:

| Method | 25% | 37.5% | 50% |
|---|---:|---:|---:|
| Age | 3 | 2 | 1 |
| Pure CNN | 15 | 7 | 4 |
| AFVT | 3 | 2 | 1 |
| MAG-VR | 5 | 3 | 1 |

AFVT exactly preserves the Age maximum stale-run bound. Its Jain fairness is
`0.9986 / 0.9833 / 0.9995` across the core budgets. MAG-VR remains controlled,
but its maximum stale run is 1.67x Age at 25% and 1.5x at 37.5%. Pure CNN shows
the expected starvation failure.

## F. Motion-Conditioned Result

AFVT is not a high-motion-only improvement:

- At 25%, it improves low and medium motion but is slightly worse on high
  motion by `+0.000049`.
- At 37.5%, it improves all three videos; the high-motion gain is largest at
  `-0.002172`.
- At 50%, it improves all three videos; the high-motion gain is again largest.

MAG-VR has the same qualitative pattern: it loses slightly on high motion at
25%, then wins all motion groups at 37.5% and 50%. The visual signal is most
valuable on the high-motion video at the two larger core budgets, but it also
provides real low- and medium-motion gains. MAG-VR does not simply collapse to
Age on low motion, although its guard consumes most of the budget when the
age boundary is active.

## G. Conditional Visual Compute

Mean CNN-evaluated blocks per frame:

| Method | 25% | 37.5% | 50% |
|---|---:|---:|---:|
| Pure CNN | 16.000 | 16.000 | 16.000 |
| AFVT | 0.766 | 6.298 | 0.340 |
| MAG-VR | 1.241 | 10.326 | 0.340 |

AFVT uses the CNN on `6.38% / 100% / 2.13%` of frames. Relative to Pure CNN,
it reduces evaluated blocks by about `95.2% / 60.6% / 97.9%`. MAG-VR reduces
them by about `92.2% / 35.5% / 97.9%`.

At 37.5%, AFVT's mean tie group is 6.298 blocks. MAG-VR allocates a mean of
4.511 blocks to its forced set and 1.589 blocks to visual selection, with a
1.42% forced-overflow rate.

## H. Profiling

Representative 37.5% selector latency on RTX 4090:

| Method | Input preparation ms | Model forward ms | Ranking ms | Total ms |
|---|---:|---:|---:|---:|
| Age | 0.149 | 0.000 | 0.083 | 0.059 |
| Pure CNN | 44.176 | 0.462 | 0.024 | 52.799 |
| AFVT | 26.385 | 0.447 | 0.085 | 38.404 |
| MAG-VR | 39.797 | 0.469 | 0.084 | 39.582 |

Stage means and total means were measured in separate runs, so they need not
sum exactly. The large mean/P95 variance comes from CPU patch preparation and
rare conditional CNN calls. The model forward remains below 0.5 ms at 37.5%;
input preparation is the dominant deployment blocker.

## I. Answers to Q1-Q9

### Q1. Why is Age strong?

Round 2.5 is reproduced. Cache age is a useful stale-lifetime prior, and the
sequential Age policy also imposes round-robin-like anti-starvation behavior.
Pure local risk ranking does not provide that long-horizon cache control.

### Q2. Does AFVT improve Age through visual tie-breaking?

Yes. AFVT beats Age at all three pooled core budgets, has strict negative
bootstrap CIs at 37.5% and 50%, wins all videos at those two budgets, and
preserves Age's maximum stale-run bounds exactly.

### Q3. Does MAG-VR prevent starvation while using visual information?

Yes, within the frozen fairness guard. It beats Age at all three pooled core
budgets and avoids Pure CNN's severe starvation. Its fairness is weaker than
AFVT, with maximum stale runs of 5 and 3 at the two lower budgets.

### Q4. Which method is most stable across the core budgets?

AFVT. Both new methods satisfy STRONG GO, but AFVT has lower average core
damage and exact Age-level starvation control. MAG-VR's small 37.5% advantage
does not offset its weaker 25% result and fairness.

### Q5. Do secondary metrics support the primary metric?

Mostly, but not completely. KL improves at every core budget. Pixel
Disagreement improves at 25% and 50% but triggers the required warning at
37.5%. This conflict is explicitly retained.

### Q6. Where does visual information help by motion regime?

The strongest gains occur on high motion at 37.5% and 50%, but the effect is
not exclusive to high motion. Low and medium motion also improve, especially
at 25% and 37.5%.

### Q7. Do the new methods reduce CNN work?

Yes. Candidate-only evaluation substantially reduces processed blocks,
especially for AFVT at 25% and 50%. Conditional compute is an algorithmic
property of the implemented schedulers, not a post-hoc optimization.

### Q8. Is selector overhead at or below 2 ms/frame?

No. AFVT is `38.404 ms/frame` and MAG-VR is `39.582 ms/frame` at 37.5%.
Neither is deployment-ready in the current CPU patch-preparation pipeline.

### Q9. Is there enough evidence to lock the final algorithm?

Yes, scientifically. Lock AFVT as the final scheduling algorithm. Do not begin
Real Partial Computation yet because the selector pipeline fails the internal
latency readiness line.

## J. Final Boundary

```text
Scientific method works.
Current preprocessing implementation is not deployment-ready.
```

Authorized next research topic:

```text
GPU-Resident Selector Pipeline Optimization
```

This report does not authorize or execute Real Partial Computation, CUDA
kernels, TensorRT, ONNX, Jetson, HiSilicon deployment, or a new scheduler.
