# DriftGuard Round 7.6 Progress

## 2026-08-25 - Initialization

```text
phase: source-of-truth and environment audit
dataset: READY on local and cloud
architecture: SegFormer-B0
source weights: /root/autodl-tmp/models/segformer-b0
source taxonomy: ADE20K 150-class initialization only
target taxonomy: VSPW 124 classes, ignore 255
temporal reuse: disabled and untouched
next step: pipeline tests and 100-step smoke
```

## 2026-08-25 - Pipeline Smoke

```text
tests: 6 passed
steps: 100 + 10-step real resume continuation
batch: 8
precision: BF16
loss: finite, approximately 4.82 -> 4.78 during warmup
peak allocated VRAM: 4.53 GB
checkpoint save/resume: PASS
decision: smoke GO
```

## 2026-08-25 - Calibration

```text
batch candidates: 2 / 4 / 6 / 8, all stable
selected batch: 8
effective batch: 8
500-step throughput: 22.98 images/s
mean step time: 0.3482 s
peak allocated VRAM: 4.53 GB
loss at step 500: approximately 3.34
estimated 40k pure training: 3.87 hours
decision: 10k pilot GO
```

## 2026-08-25 - Pilot Start

```text
tmux session: driftguard-r76
target: 10,000 optimizer steps
validation: steps 5,000 and 10,000
schedule horizon: 40,000 steps
temporal reuse: not executed
```
