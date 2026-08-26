# Claim Audit

## Allowed claims

- Whole-Stage4 temporal reuse can skip a dense Stage 4 forward in the tested SegFormer-B0/B1 pipelines.
- B0 FI-2 has a repeated paired mean speedup of 1.0924x with a small official validation mIoU drop of 0.0919 percentage points.
- B1 preserves quality at FI-2 but has weak and statistically uncertain speed gain; FI-8 exposes a larger quality/latency trade-off.
- Stage 4 runtime share limits the system benefit through Amdahl's law.
- The separately scoped Round 5 B0 Stage4 microbenchmark measured 4.366 ms for dense Stage4 and 6.662 ms for the 37.5% cached-context route.

## Prohibited claims

- State-of-the-art performance.
- Generalization to all SegFormer models or all video datasets.
- Independent VSPW test-set accuracy.
- Edge-device validation or real-time guarantees.
- Speedup from block-sparse or partial attention computation.
- Cityscapes cross-dataset results.
- A causal or universal relationship between Stage4 share and model scale.
