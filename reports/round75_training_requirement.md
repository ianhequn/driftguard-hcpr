# DriftGuard Round 7.5 Training Requirement

## Status

```text
MODEL STATUS: NOT FOUND
FUTURE TRAINING: REQUIRED IF NO COMPATIBLE PUBLIC CHECKPOINT IS RECOVERED
TRAINING EXECUTION: NOT AUTHORIZED IN ROUND 7.5
```

The official VSPW code and baseline repositories did not expose a verified
VSPW-trained SegFormer checkpoint. The current ADE20K checkpoint is
incompatible and must not be fine-tuned in this round.

## Future Training Plan

```text
architecture: SegFormer family, preferably B0 if resource and accuracy
              permit; B1/B2 only with explicit scale-change record
dataset: official VSPW train split
model selection: official VSPW validation split
classes: 124 valid VSPW classes
ignore index: 255 after official mask normalization
loss: standard semantic segmentation loss with ignore_index=255
checkpoint rule: highest validation mIoU with reproducible seed/config
reuse benchmark: only after checkpoint is frozen and compatibility audited
```

This is a planning artifact, not a training run. No GPU hours, expected mIoU,
or storage estimate is claimed until the official package and model
configuration are fixed.
