# Round 7 Restart: Accuracy / Latency Pareto

The primary plot is `results/R7R_E05_pareto/accuracy_latency_pareto.png`. The companion plot is `miou_drop_vs_speedup.png`; the tabular source is `main_results.csv`.

## Main operating points

| Policy | mIoU | Drop (pp) | Speedup | Interpretation |
|---|---:|---:|---:|---|
| Full | 0.301292 | 0.0000 | 1.000x | quality reference |
| FI-2 | 0.300373 | 0.0919 | 1.232x | primary strong point |
| FI-3 | 0.298768 | 0.2524 | 1.042x | small measured speedup |
| FI-4 | 0.296928 | 0.4364 | 0.868x | slower in paired run |
| FI-6 | 0.292336 | 0.8956 | 1.119x | speed/quality trade-off |
| FI-8 | 0.289973 | 1.1319 | 1.288x | faster, larger damage |

## Bootstrap

Video-level bootstrap used 10,000 resamples with seed 42. For FI-2 minus Full mIoU, the mean delta was `0.0035818` in the script's per-video aggregation convention, with 95% interval `[0.0023861, 0.0047994]`. The primary decision still uses the aggregate pixel-confusion mIoU drop reported in the main table; the bootstrap is an uncertainty diagnostic, not a replacement metric.

## Decision

FI-2 occupies the required practical Pareto region: it retains nearly the Full ground-truth mIoU while providing more than 10% real paired end-to-end acceleration. FI-8 is faster but has a larger quality loss. No claim is made beyond this frozen validation split.
