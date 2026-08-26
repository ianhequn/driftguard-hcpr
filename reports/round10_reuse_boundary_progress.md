# Round 10 Progress

## Final status

Round 10 is complete on the replacement SSH endpoint (port 12246). The formal
host passed the hardware gate: NVIDIA GeForce RTX 4090 D, driver 595.71.05,
PyTorch 2.1.2+cu118, CUDA 11.8, and Transformers 4.44.2.

## Completed before the gate

- Created branch `research/round10-reuse-boundary` from a clean worktree.
- Audited the frozen B0/B1 checkpoint identities, VSPW validation contract,
  prior Stage4 dependency result, and existing claim/reproducibility files.
- Added `StructuredBoundaryReusePipeline` for exactly S4, S3-S4, and S2-S4.
- Added physical stage-call counters and persistent cache-byte accounting.
- Added CPU-level semantic tests for full refresh, skipped suffixes, cache
  replacement, and decoder-per-frame behavior.
- Added `scripts/run_round10_preflight.py`; the cloud preflight passes and is
  stored under `results/R10_E00_preflight/preflight.json`.

## Completed

- B0/B1 100% refresh equivalence passed for all three boundaries.
- The 20-configuration two-video smoke test passed.
- Full VSPW validation quality matrix: 20/20 configurations.
- Stage profile: 3 repetitions with 200 warmup and 1,000 measurements.
- Paired latency: 18 reuse configurations, 5 repetitions, 200 warmup and
  1,000 measurements per configuration, with immediate paired Full baselines.
- Semantic and reuse test suite: 29 passed.
- Final decision: **ROUND 10 = STRONG BOUNDARY RESULT**.

Historical RTX 4090 values remain context only and are not substituted for the
new RTX 4090 D measurements.

## Final stop

Do not open new datasets, model scales, boundaries, adaptive selectors, lower-
level kernels, or deployment backends in Round 10. Return to paper writing.
