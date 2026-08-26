# Claim Audit

## Allowed claims

- Whole-Stage4 temporal reuse can skip a dense Stage 4 forward in the tested SegFormer-B0/B1 pipelines.
- B0 FI-2 has a repeated paired mean speedup of 1.0924x with a small official validation mIoU drop of 0.0919 percentage points.
- B1 preserves quality at FI-2 but has weak and statistically uncertain speed gain; FI-8 exposes a larger quality/latency trade-off.
- Stage 4 runtime share limits the system benefit through Amdahl's law.

## Prohibited claims

- State-of-the-art performance.
- Generalization to all SegFormer models or all video datasets.
- Independent VSPW test-set accuracy.
- Edge-device validation or real-time guarantees.
- Speedup from block-sparse or partial attention computation.
- Cityscapes cross-dataset results.
