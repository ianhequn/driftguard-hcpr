# DriftGuard Round 8 Resume Progress

日期：2026-08-25

## Completed

1. Preserved the historical Round 8 source-blocked attempt.
2. Confirmed `hf-mirror.com` as the download endpoint while retaining the original Hugging Face source identity.
3. Recovered the complete B1 source snapshot.
4. Resolved revision and recorded file hashes.
5. Performed fresh source load and CUDA forward smoke.
6. Passed the source recovery gate.
7. Constructed the VSPW-124 SegFormer-B1 model and passed the architecture audit:
   - ADE20K source classifier removed and VSPW-124 classifier reinitialized.
   - F1-F4 shapes: `[1,64,128,128]`, `[1,128,64,64]`, `[1,320,32,32]`, `[1,512,16,16]`.
   - Decoder output: `[1,124,128,128]` with finite values.
   - Whole-Stage4 reuse compatibility: `DIRECT`.
8. Completed a 100-step BF16 smoke run and checkpoint reload:
   - effective batch size 8, finite loss, reload check passed.
9. Completed batch calibration and selected physical batch 8:
   - 500-step mean step time: `0.3733934290 s`.
   - throughput: `21.4251 images/s`.
   - peak VRAM: `5.4146 GB`.
   - estimated pure 40k-step training time: approximately `4.15 h`, excluding validation and checkpoint I/O.

## Current status

```text
ROUND 8 SOURCE RECOVERY = PASS
ROUND 8 B1 INITIALIZATION = PASS
ROUND 8 B1 CALIBRATION = PASS
ROUND 8 RESUME = AUTHORIZED
```

An independent full-validation timing pass is running before clean 40k training. Round 7.1 will not be rerun.
