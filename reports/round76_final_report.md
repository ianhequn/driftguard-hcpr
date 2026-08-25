# DriftGuard Round 7.6 Final Report

## Final Decision

```text
ROUND 7.6:
MODEL TRAINING NO-GO

MODEL:
SegFormer-B0 VSPW-124 (attempted, not frozen)

FROZEN CHECKPOINT:
NONE

CHECKPOINT SHA256:
NOT AVAILABLE

VALIDATION mIoU:
NOT AVAILABLE

BEST STEP:
NONE

TRAINING GPU HOURS:
0.5183 active training / 0.6366 total process

CLASS ALIGNMENT:
124 / 124 VERIFIED FOR THE CONSTRUCTED MODEL
FROZEN-CHECKPOINT GATE NOT REACHED

STRUCTURED REUSE COMPATIBILITY:
NOT AUDITED - NO FROZEN MODEL

BENCHMARK READINESS:
NOT READY
```

Round 7.6 stopped at the mandatory dataset-integrity boundary. No temporal
reuse, FI-2, latency benchmark, checkpoint freeze, or Round 7 restart was run.

## Blocking Finding

The first full validation at step 5,000 failed because a normalized GT value
`252` entered the 124-class confusion matrix. Exhaustive raw-mask scanning then
established the cause:

| Split | Masks scanned | Anomalous masks | Invalid pixels | Invalid raw ID |
|---|---:|---:|---:|---:|
| Train | 197,253 | 0 | 0 | None |
| Validation | 24,392 | 17 | 68,314 | 253 |

All anomalies are consecutive frames `00000391.png` through `00000407.png` in
validation video `353_-5sqmCF8OFU`. The first affected mask has identical
local/cloud SHA256:

```text
8d0d7eb3d334f08d6fce9676199bb1d41ecf1b0b7cfbe740a49d635c6c08acac
```

The official download documentation declares only raw `0..124` and `255` and
recommends the frozen normalization. The official baseline code applies that
normalization, while its evaluator separately keeps only normalized GT IDs in
`[0, 124)`. That range filter would silently exclude normalized `252`.

Round 7.6 could not adopt this as an unplanned ignore rule: the task freezes
`255` as the sole loss/evaluation ignore ID and explicitly requires stopping
when dataset integrity or audited label semantics fail. Continuing would first
require a new, explicit dataset/evaluation contract.

Official primary-source references:

```text
https://github.com/VSPW-dataset/VSPW-dataset-download
https://github.com/VSPW-dataset/VSPW_baseline/blob/master/dataset.py
https://github.com/VSPW-dataset/VSPW_baseline/blob/master/utils.py
```

## Training Evidence

The model itself showed healthy optimization before the hard stop:

```text
start: 2026-08-25 09:13:52 +08:00
end: 2026-08-25 09:52:04 +08:00
steps: 5,000
images seen: 40,000
epoch equivalent: 0.2028
loss at step 10: 4.8205
loss at step 5,000: 1.2041
NaN / Inf: none observed
peak allocated VRAM: 4.53 GB
active training: 1,865.7 s / 0.5183 h
attempted validation: approximately 0.113 h
total process: 0.6366 h
```

The original runner performed validation before its periodic 5k save, so the
validation exception occurred before any pilot checkpoint was written. The
runner is now hardened to save a periodic state before validation, and the
loader/metric raise explicit label-domain errors. Post-fix cloud tests: `8
passed`.

## Required Answers

**Q1. Initialization:** ADE20K-style SegFormer-B0 at
`/root/autodl-tmp/models/segformer-b0`, SHA256
`6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9`.

**Q2. Reinitialized parameters:** `decode_head.classifier.weight` and
`decode_head.classifier.bias`; 206 compatible tensors loaded, no unexpected
keys.

**Q3. Batch:** per-GPU 8, effective 8, no accumulation.

**Q4. Precision:** BF16, finite throughout the 5k attempt.

**Q5. Calibration:** 0.3482 s/step and 22.98 images/s at batch 8.

**Q6. GPU cost:** 0.5183 active training hours, approximately 0.113 attempted
validation hours, 0.6366 total process hours.

**Q7. Training length:** 5,000 optimizer steps, 40,000 images, 0.2028 epoch
equivalent.

**Q8. Best step:** none; no valid validation checkpoint existed.

**Q9. Best validation mIoU:** unavailable; the first full pass was invalidated.

**Q10. Frozen reload:** not run because no model was frozen.

**Q11. Output classes:** the constructed runtime model strictly output 124
channels; no frozen artifact exists.

**Q12. Class order:** pipeline mapping and tests verified 124/124; the
frozen-checkpoint alignment gate was not reached.

**Q13. Failures:** no NaN/Inf was observed and the loss fell normally. Class
collapse could not be audited. A real official-package label-domain bug was
confirmed in validation.

**Q14. Structured reuse compatibility:** not formally audited because the task
permits this only after freeze. The attempted architecture remained standard
SegFormer-B0, but no compatibility claim is issued.

**Q15. Round 7 restart:** not ready. It lacks a selected, frozen, reloaded
checkpoint and a valid full-validation mIoU.

## Retained Artifacts

```text
results/R76_E01_pipeline_smoke/label_domain_summary.json
results/R76_E01_pipeline_smoke/label_domain_anomalies.csv
results/R76_E03_training/pilot.log
results/R76_E03_training/run_config.json
results/R76_E03_training/training_history.csv
results/R76_E03_training/training_loss_curve.png
results/R76_E03_training/learning_rate_curve.png
results/R76_E03_training/validation_miou_curve.png
```

No failed checkpoint exists because the failure preceded the original 5k save.
No model weights, optimizer state, VSPW data, or large predictions are committed.
