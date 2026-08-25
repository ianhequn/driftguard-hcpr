# DriftGuard Round 7 Restart Final Report

日期：2026-08-25

## Final decision

```text
ROUND 7 RESTART = STRONG GO
```

The pre-declared primary policy FI-2 achieves both required conditions on the frozen VSPW validation benchmark:

- real paired CUDA-resident end-to-end speedup: **1.232x**;
- ground-truth mIoU drop: **0.0919 percentage points**.

This is a real measured whole-Stage4 reuse result. It is not a sparse-compute, FLOPs-only, selector-only or Stage4-only claim.

## Experimental contract

- Frozen checkpoint: `segformer-b0-vspw124`, SHA256 `8b673cb6bc173f3e986da8446c2c3debacc77779109b4916b230af36f197f17b`.
- Frozen official VSPW validation split: 343 videos, 24,392 RGB-GT pairs, 124 classes.
- Full reference mIoU: `0.30129196806407454`; measured Full mIoU: `0.30129196806407454`.
- Stage1-3 are fresh on every frame; Stage4 is dense refresh or cached final F4; decoder runs on every frame.
- First frame refreshes and cache resets at every video boundary. Processing is chronological and unshuffled.
- Policies are limited to Full, FI-2, FI-3, FI-4, FI-6 and FI-8. FI-2 was fixed as primary before reading its result.

The public VSPW test split has no public ground truth, so both checkpoint selection and this GT benchmark use the validation split. Accordingly, this report establishes a validation-system result, not independent test-set generalization.

## Ground-truth quality

| Policy | mIoU | Drop (pp) | Pixel accuracy | Actual refresh | Stage4 calls |
|---|---:|---:|---:|---:|---:|
| Full | 0.301292 | 0.0000 | 0.703089 | 100.000% | 24,392 |
| FI-2 | 0.300373 | 0.0919 | 0.702262 | 50.381% | 12,289 |
| FI-3 | 0.298768 | 0.2524 | 0.700828 | 33.691% | 8,218 |
| FI-4 | 0.296928 | 0.4364 | 0.699142 | 25.508% | 6,222 |
| FI-6 | 0.292336 | 0.8956 | 0.695309 | 17.264% | 4,211 |
| FI-8 | 0.289973 | 1.1319 | 0.691605 | 13.041% | 3,181 |

FI-2 passes the quality gate by a wide margin. The full curve was still measured because no early quality stop was triggered.

## Equivalence and counter evidence

On 20 validation frames at 100% refresh, max absolute error, mean absolute error, relative L2 and prediction disagreement were all zero. The formal Full run executed Stage4 24,392 times; FI-2 executed it 12,289 times while the decoder still processed all 24,392 frames. This confirms that reuse skipped the dense Stage4 forward rather than masking a full result after the fact.

## Latency

The controlled protocol used 200 warmup and 1,000 measurements with CUDA Events and an adjacent paired Full run for each policy. Main rows:

| Policy | Mean ms | Median ms | P95 ms | P99 ms | FPS | Paired speedup |
|---|---:|---:|---:|---:|---:|---:|
| Full | 14.323 | 14.153 | 19.511 | 36.859 | 69.82 | 1.000x |
| FI-2 | 12.185 | 11.654 | 15.070 | 16.382 | 82.07 | 1.232x |
| FI-3 | 12.390 | 11.771 | 15.239 | 16.894 | 80.71 | 1.042x |
| FI-4 | 15.913 | 11.886 | 43.733 | 61.889 | 62.84 | 0.868x |
| FI-6 | 11.409 | 11.556 | 14.893 | 16.282 | 87.65 | 1.119x |
| FI-8 | 10.868 | 10.955 | 14.196 | 15.481 | 92.01 | 1.288x |

The primary FI-2 result is therefore a genuine end-to-end speedup. The non-monotonic FI-4 row is retained as a GPU timing observation, not hidden by averaging with other budgets.

## Compute economics and memory

The component profile measured Stage4 at 1.829 ms out of 10.428 ms, or 17.536% of the profiled Full path. The zero-Stage4 Amdahl upper bound was 1.213x. The measured system result is consistent with the fact that only whole Stage4 is skipped while Stage1-3 and decoder remain active. The persistent final-F4 cache is 0.125 MB in BF16; latency-run peak allocated GPU memory was 0.8568 GB.

## Bootstrap and per-class evidence

The video-level bootstrap contains 10,000 resamples with seed 42. Full/FI-2 per-class IoU, support and deltas are in `results/R7R_E05_pareto/full_vs_fi2_per_class.csv`. The frozen checkpoint's long-tail failures from Round 7.6 remain: classes 43, 47, 95 and 113 were never predicted, with sparse support also affecting classes 4, 72 and 73. These limitations are reported rather than attributed to temporal reuse.

## Historical boundaries and stop point

Round 5 block-sparse partial computation remains NO-GO. Round 6 dynamic visual-risk/stage-age policies remain NO-GO because selector overhead erased their benefit. Round 7 Restart validates only fixed whole-Stage4 reuse on this frozen VSPW validation split. No VSPW test-set claim, new dataset, new checkpoint, multi-stage reuse, custom kernel, TensorRT, edge-device port or paper-writing phase is started here.

## Reproducibility artifacts

- `results/R7R_E00_preflight/equivalence.json`
- `results/R7R_E02_full_gt/metrics.json`
- `results/R7R_E03_fi2_gate/metrics.json`
- `results/R7R_E04_full_curve/FI-3/metrics.json` through `FI-8/metrics.json`
- `results/R7R_E05_pareto/main_results.csv`
- `results/R7R_E05_pareto/video_bootstrap.csv`
- `results/R7R_E05_pareto/full_vs_fi2_per_class.csv`
- `results/R7R_E06_latency/latency_summary.json`
- `results/R7R_E06_latency/stage_profile.json`

## Verification

云端 `/root/autodl-tmp/DriftGuard` 使用项目 Python 环境执行完整测试：`94 passed, 2 skipped`。本机仅完成 Python `compileall`，因为本机没有安装 pytest；因此测试结论以云端环境为准。
