# DriftGuard Round 7.1 Final Report

日期：2026-08-25

## Final output

```text
ROUND 7.1 = ROBUST GO
FI-2 REPEATED SPEEDUP = 1.0924 ± 0.0757x
FI-2 95% CI = [1.0337, 1.1510]
FI-2 MIN / MAX = 1.0068x / 1.1783x
FI-4 STATUS = TRANSIENT OUTLIER
IMPLEMENTATION = VALID
ROUND 8 LONG TRAINING = AUTHORIZED
```

## Answer to the primary question

Yes. The B0 FI-2 acceleration is robust across five independent paired CUDA measurements under the frozen execution contract. The result is a robust real speedup, but not a `ROBUST STRONG GO` under the project's internal 1.10 mean-speedup line.

The frozen Round 7 GT quality result remains unchanged: Full mIoU `0.30129196806407454`, FI-2 mIoU `0.3003732475378488`, drop `0.0918721 pp`. Round 7.1 did not rerun that benchmark.

## Environment and correctness

The frozen B0 checkpoint SHA256 matched exactly. The RTX 4090 environment reported 48°C at the gate, 1 MiB memory use, 210 MHz SM clock and zero utilization; no competing CUDA compute app was present. Stage1-3 remained fresh, Stage4 was dense-refresh or cached-final-F4, decoder ran every frame, and video cache reset semantics were retained by the runner.

## Profile audit

Three independent profiles measured:

| Component | Mean ms | Across-repetition std ms |
|---|---:|---:|
| Full | 11.328 | 0.394 |
| Stage1-3 | 7.865 | 0.633 |
| Stage4 | 2.716 | 0.376 |
| Decoder | 0.984 | 0.152 |

The profile-derived Stage4 fraction is `23.975%`; its zero-Stage4 decomposition gives an indicative Amdahl value of approximately `1.315x`. This is not a strict upper-bound comparison against the paired end-to-end values because the profile and paired benchmark are different measurement decompositions. The report keeps that distinction explicit.

## FI-4 anomaly

The Round 7 single-run FI-4 slowdown was not reproduced. All five independent FI-4 paired speedups exceeded 1.0, so the correct classification is `TRANSIENT OUTLIER`, not a stable performance pathology. This does not erase the original observation; it narrows its interpretation.

## Serial authorization

The technical gate for Round 8 is satisfied because the implementation is valid and FI-2 mean speedup is greater than 1.0. GPU timing jobs have completed and the GPU is released before B1 initialization. Round 8 must keep the frozen VSPW split, label contract, whole-Stage4 reuse semantics and B1-only scope.

## Artifacts

- `results/R71_E00_environment/environment.json`
- `results/R71_E01_repeated_latency/repeated_latency.csv`
- `results/R71_E01_repeated_latency/repeated_latency.json`
- `results/R71_E02_profile/profile_repetitions.json`
- `results/R71_E03_analysis/decision.json`
- `results/R71_E03_analysis/repeated_speedup.png`
