# Round 7.1 Latency Robustness Audit

## Frozen contract

- Model: SegFormer-B0 VSPW-124
- Model SHA256: `8b673cb6bc173f3e986da8446c2c3debacc77779109b4916b230af36f197f17b`
- Precision: BF16
- Input: CUDA-resident `1 x 3 x 512 x 512`
- Warmup: 200
- Measurements: 1000 per policy and repetition
- Repetitions: 5
- Paired Full follows each FI measurement immediately
- Tensor pool, seed and seeded policy permutations are shared across repetitions

No GT benchmark was rerun in Round 7.1.

## Repeated speedup summary

| Policy | Mean speedup | Std | CV | Min | Max | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| FI-2 | 1.0924x | 0.0757 | 0.0693 | 1.0068x | 1.1783x | [1.0337, 1.1510] |
| FI-3 | 1.1077x | 0.0757 | 0.0684 | 1.0216x | 1.2113x | [1.0504, 1.1683] |
| FI-4 | 1.1780x | 0.1266 | 0.1075 | 1.0699x | 1.3817x | [1.0907, 1.2839] |
| FI-6 | 1.2023x | 0.0428 | 0.0356 | 1.1595x | 1.2707x | [1.1724, 1.2369] |
| FI-8 | 1.2134x | 0.1732 | 0.1427 | 1.0896x | 1.5106x | [1.1068, 1.3680] |

The CI is a bootstrap percentile interval over the five independent repetition means, seed 42. The authoritative raw table is `results/R71_E01_repeated_latency/repeated_latency.csv`.

## FI-2 gate

FI-2 is the primary policy. It satisfies `mean speedup > 1.0` and `CI95 lower > 1.0`, so it is a **ROBUST GO**. It does not satisfy the internal `mean >= 1.10` Strong GO threshold.

## FI-4 audit

Round 7's single-run FI-4 result was `0.8684x` with P95 `43.733 ms`. In Round 7.1, all five FI-4 repetition speedups were above 1.0: `[1.3817, 1.0794, 1.0699, 1.2041, 1.1550]`. The old low-speedup row therefore was not reproduced as a stable pathology and is classified as:

```text
FI-4 = TRANSIENT OUTLIER
```

The old observation remains documented and was not deleted.

## Validity

The model hash, environment gate, five repetitions, warmup/measurement counts, BF16 precision, policy coverage, paired Full fields and three-repetition profile checks all passed. No competing compute process was present; Jupyter and TensorBoard were idle service processes.
