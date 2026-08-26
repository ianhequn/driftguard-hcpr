# DriftGuard Round 8 Resume: B1 Latency

Protocol: CUDA-resident B=1, 512x512, BF16, 200 warmup frames, 1000 measured frames, same frozen input pool, paired Full immediately after each reuse run, three independent repetitions.

| Policy | Mean Speedup | Std | 95% CI | Min | Max | Mean Policy ms | Mean P95 ms | Mean P99 ms |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| FI-2 | 1.0203x | 0.0679 | [0.8515, 1.1891] | 0.9806 | 1.0987 | 13.454 | 27.715 | 40.521 |
| FI-4 | 1.0413x | 0.1571 | [0.6509, 1.4317] | 0.8877 | 1.2018 | 13.465 | 23.594 | 38.536 |
| FI-8 | 1.2245x | 0.1591 | [0.8292, 1.6199] | 1.0856 | 1.3981 | 11.015 | 14.342 | 22.505 |

The wide intervals reflect real GPU timing variability with only three repetitions; they are reported rather than hidden. FI-8 is the clear speed/quality trade-off point, while the primary FI-2 point has only weak positive mean speedup.

The component profile used three repetitions of 200 warmups and 1000 measurements. Mean Stage4 fraction was `18.99%` of the isolated full forward. Stage4 F4 shape was `[1,512,16,16]`, BF16, with `0.25 MB` persistent cache.
