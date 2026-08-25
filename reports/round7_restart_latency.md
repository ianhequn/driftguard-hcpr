# Round 7 Restart: CUDA Latency and Compute Profile

## Protocol

Input boundary is a CUDA-resident preprocessed tensor, batch size 1, BF16 inference. Each policy uses 200 warmup iterations and 1,000 CUDA Event measurements over the same frozen VSPW validation tensor pool. Each fixed-interval policy is followed immediately by a paired Full measurement in the same process. The reported speedup is `paired_full_mean / policy_mean`. Disk I/O, decoding, pageable H2D, visualization and CSV writing are excluded.

## Paired end-to-end latency

| Policy | Mean ms | Median ms | P95 ms | P99 ms | FPS | Paired Full ms | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 14.323 | 14.153 | 19.511 | 36.859 | 69.82 | 14.323 | 1.000x |
| FI-2 | 12.185 | 11.654 | 15.070 | 16.382 | 82.07 | 15.016 | 1.232x |
| FI-3 | 12.390 | 11.771 | 15.239 | 16.894 | 80.71 | 12.905 | 1.042x |
| FI-4 | 15.913 | 11.886 | 43.733 | 61.889 | 62.84 | 13.819 | 0.868x |
| FI-6 | 11.409 | 11.556 | 14.893 | 16.282 | 87.65 | 12.765 | 1.119x |
| FI-8 | 10.868 | 10.955 | 14.196 | 15.481 | 92.01 | 14.000 | 1.288x |

FI-2 is the pre-declared primary point and has a real paired end-to-end speedup of `1.232x`. FI-4 is slower in this run, which is retained as measured rather than smoothed away. The paired protocol is authoritative; non-paired historical latency values are not reused.

## Stage profile and Amdahl ceiling

The component profile used 200 warmup and 1,000 measurements. Mean Full was 10.428 ms, Stage1-3 was 6.815 ms, Stage4 was 1.829 ms, and Decoder was 0.817 ms. Stage4 represented 17.536% of the profiled Full path. If Stage4 were reduced to zero at unchanged other costs, the Amdahl upper bound would be `1.213x`. This is an engineering ceiling for this profile, not a general threshold.

The timing artifact reports a persistent final-F4 cache of `0.125 MB` in BF16 and peak allocated GPU memory of `0.8568 GB` during the latency run. No additional cache larger than 1 GB was introduced.

## Measurement boundary

The formal ground-truth quality runner includes model execution, cache transitions and decoder for each frame, while the latency runner isolates the CUDA-resident model boundary required for controlled paired timing. Neither result includes video decoding or file I/O.
