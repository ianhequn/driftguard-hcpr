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

## 2026-08-25 - Pilot Hard Stop

```text
phase: 5k full validation
steps completed: 5,000
images seen: 40,000
epoch equivalent: 0.2028
loss: finite, approximately 4.82 -> 1.20
active training time: 1,865.7 s (0.5183 h)
attempted validation time: approximately 0.113 h
total process time: 0.6366 h
peak allocated VRAM: 4.53 GB
bug: validation mask normalized to class ID 252 and broke confusion reshape
checkpoint: NONE; the original runner validated before its 5k save
decision: automatic dataset-integrity hard stop
next step: full label-domain audit and NO-GO report
```

## 2026-08-25 - Full Label-Domain Audit

```text
train masks scanned: 197,253
train anomalies: 0
validation masks scanned: 24,392
validation anomalies: 17 masks / 68,314 pixels
raw invalid ID: 253
affected video: 353_-5sqmCF8OFU
affected frames: 00000391.png through 00000407.png
local/cloud first-mask SHA256: 8d0d7eb3d334f08d6fce9676199bb1d41ecf1b0b7cfbe740a49d635c6c08acac
official download contract: raw 0..124 and 255 only
official baseline evaluator: additionally filters normalized GT to 0 <= ID < 124
project contract: 255 is the sole ignore ID; changing it is forbidden in this run
tests after fail-fast hardening: 8 passed
final decision: MODEL TRAINING NO-GO
```
