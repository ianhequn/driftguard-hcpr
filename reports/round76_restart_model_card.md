# SegFormer-B0 VSPW-124 Restart Model Card

This is a research baseline trained on the frozen VSPW official train/val
split. It uses the Round 7.65 canonical label contract and is intended for the
next separately controlled ground-truth benchmark only after the restart gate
passes.

- Input: RGB, resized to 512 x 512, ImageNet normalization.
- Output: 124 semantic logits; output IDs are 0..123.
- Target ignore: 255.
- Initialization: ADE20K-style SegFormer-B0, classifier-only reinitialization.
- Training: 40000 steps, batch 8, BF16, seed 42.
- Selected checkpoint: step 40000, validation mIoU 0.3012895794.
- Fresh-process validation: mIoU 0.3012919681; absolute delta 2.39e-6.
- Temporal reuse: not part of this artifact or its validation.
- Dataset files: not modified.

This card does not claim temporal-reuse accuracy, latency, or generalization
beyond the frozen VSPW validation protocol.
