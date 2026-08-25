# DriftGuard Round 7.6 Preprocessing Audit

## Frozen Contract

```text
input: RGB
training resolution: 512 x 512
validation resolution: 512 x 512
image resize: bilinear
mask resize: nearest-neighbor
image mean: 0.485 / 0.456 / 0.406
image std: 0.229 / 0.224 / 0.225
training augmentation: random horizontal flip p=0.5
validation augmentation: none
```

Round 6 also used the historical 512x512 SegFormer processor. Its processor
contains ADE-style `reduce_labels=true`; Round 7.6 does not delegate GT
normalization to that processor. The dataset loader explicitly applies the
audited VSPW rule exactly once:

```python
label[label == 0] = 255
label = label - 1
label[label == 254] = 255
```

The resulting valid IDs are `0..123`; `255` is excluded by both loss and mIoU.
Ten RGB images and ten masks decoded successfully before training. Sampled
source dimensions were predominantly `853x480`, with one valid `854x480`
video. RGB and mask dimensions matched within every sampled video.

Validation is deterministic and uses the official `val.txt`; no train,
validation, or test split is regenerated.
