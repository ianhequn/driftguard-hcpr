# DriftGuard Round 7.1 Progress

日期：2026-08-25

## Scope

Round 7.1 只审计冻结 SegFormer-B0 VSPW-124 的 latency robustness，不重跑 GT quality，不改模型、precision、FI semantics、输入分辨率或 reuse implementation。

## Completed

1. Model SHA256 and RTX 4090 environment gate passed.
2. Five independent repetitions completed for Full, FI-2, FI-3, FI-4, FI-6 and FI-8.
3. Each FI used 200 warmup + 1000 CUDA Event measurements followed immediately by paired Full measurement.
4. Fixed seeded policy permutations were used across repetitions.
5. Three independent Stage1-3 / Stage4 / Decoder profiles completed.
6. FI-2 robustness, FI-4 anomaly and Round 8 authorization gate analyzed.

## Decision

`ROUND 7.1 = ROBUST GO`

The mean FI-2 speedup is below the internal 1.10 Strong threshold, but remains above 1.0 and its five-repetition bootstrap 95% CI is entirely above 1.0. Round 8 long training is therefore authorized by the serial gate.
