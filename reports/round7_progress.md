# DriftGuard Round 7 Progress

## 2026-08-24 - Round Start

```text
phase: source-of-truth and dataset validity gate
dataset status: VSPW not installed locally or on cloud
model status: frozen SegFormer-B0 ADE20K checkpoint found
label-space status: pending official dataset audit
decision: do not run mIoU before access and compatibility checks
next step: verify official VSPW mask semantics, source, and split metadata
```

## 2026-08-24 - Dataset and Label Audit

```text
official source: VSPW-dataset/VSPW-dataset-download
official mask semantics: values 0..124 and 255;
                         0 and 255 are void during evaluation
official normalized valid labels: 0..123
local VSPW root: absent
cloud VSPW root: absent
download probe: official 480P Google Drive endpoint did not respond
frozen checkpoint: 150 ADE20K-style output labels
compatibility: INCOMPATIBLE without an official complete mapping
```

## 2026-08-24 - Round Stop

```text
decision: ROUND 7 = MODEL-DATASET COMPATIBILITY NO-GO
ground-truth validated speedup: NO
reason: dataset access blocked and frozen checkpoint label space is not
        directly compatible with VSPW
next step: dataset-compatible frozen segmentation model benchmark only
```
