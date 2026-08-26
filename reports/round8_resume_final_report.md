# DriftGuard Round 8 Resume Final Report

## Decision

```text
ROUND 8 SOURCE RECOVERY = PASS
ROUND 8 B1 GENERALIZATION = MODEL-SCALE CONDITIONAL
```

## Why

The predeclared official B1 source was recovered through `https://hf-mirror.com` at a recorded revision while preserving the model identity `nvidia/segformer-b1-finetuned-ade-512-512`. The ADE20K 150-class head was replaced with a freshly initialized VSPW-124 head under the fixed label contract. The model trained cleanly for 40,000 steps and reached mIoU `0.3715982824`.

Whole-Stage4 reuse is semantically valid: 20 real validation frames at 100% refresh matched the standard forward with zero measured error, and fresh model reload matched the final validation within `3.51e-7` mIoU.

The primary FI-2 point preserves quality well (`0.1185 pp` global mIoU drop) but its three-repeat paired mean speedup is only `1.0203x`, with a wide 95% interval crossing 1.0. FI-8 reaches `1.2245x` mean speedup, but incurs `1.4353 pp` mIoU loss. Therefore this is a real, measurable trade-off, not a strong primary-go result.

## Bootstrap

The 10,000-sample, seed-42 video-level bootstrap is stored in `results/R8R_E10_cross_scale/bootstrap_fi2.json`. Its secondary video-mean mIoU difference is not the official global mIoU metric; the primary global drop remains `0.1185 pp`.

## Scope Stop

Round 8 stops here. No new scheduler, predictor, model, dataset split, multi-stage reuse, custom CUDA, TensorRT, edge hardware, or larger dataset benchmark is started.
