# DriftGuard Round 7.6 Model Card

## Status

```text
MODEL CARD NOT ISSUED FOR A FROZEN MODEL
ROUND 7.6 = MODEL TRAINING NO-GO
```

No checkpoint was selected or frozen. This file records the attempted model so
that the failed run cannot be mistaken for a deployable VSPW baseline.

## Attempted Model

```text
architecture: SegFormer-B0
initialization: /root/autodl-tmp/models/segformer-b0
initialization SHA256: 6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9
dataset: official VSPW 480P
training split: official train.txt
validation split: official val.txt
output classes: 124
class mapping: official raw IDs 1..124 -> model IDs 0..123
ignore index: 255
input: RGB, 512x512, ImageNet normalization
optimizer: AdamW, learning rate 6e-5, weight decay 0.01
precision: BF16
batch/effective batch: 8 / 8
completed steps: 5,000
best step: NONE
validation mIoU: NOT AVAILABLE
checkpoint SHA256: NOT AVAILABLE
```

## Blocking Limitation

Seventeen official validation masks contain raw ID `253`, contradicting the
frozen raw domain `0..124/255`. This triggered the required integrity hard stop
before checkpoint selection. The training loss was finite and falling, but no
valid full-validation metric exists and no model artifact may be used as a
Round 7 baseline.
