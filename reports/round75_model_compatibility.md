# DriftGuard Round 7.5 Model Compatibility

## Compatibility Gate

```text
selected model: NONE
label space: PARTIALLY VERIFIED FOR CONFIG-ONLY MPVSS CANDIDATE
class alignment: 0/124 verified
forward smoke: NOT RUN
structured reuse compatibility: UNKNOWN / MAJOR ADAPTATION FOR MPVSS
```

The official-author `ziplab/MPVSS` VSPW configuration is a valid
configuration-level taxonomy reference:

```text
NUM_CLASSES: 124
IGNORE_VALUE: 255
TRAIN: vspw_sem_seg_video_train
TEST: vspw_sem_seg_video_val
```

It is not a valid frozen model candidate because no VSPW checkpoint artifact
was found in the audited repository. Its Mask2Former/MPVSS architecture does
not expose the Round 6 SegFormer F1/F2/F3/F4 contract, so structured reuse
would require a major adaptation and remains outside this round.

The Round 6 ADE20K SegFormer-B0 remains:

```text
num_labels: 150
training taxonomy: ADE20K-style
```

It cannot be promoted to a VSPW-compatible model by truncating logits,
renaming classes, or manually merging categories.

## Required Alignment

A valid future candidate must provide a complete, ordered mapping:

```text
VSPW class 0 -> model class 0
...
VSPW class 123 -> model class 123
ignore index -> evaluation-only void handling
```

The candidate must also expose a real checkpoint and explicit VSPW training
provenance. A matching number of classes without class-order evidence is not
enough.

## Structured Reuse Static Assessment

No valid frozen candidate was available for architecture inspection. The
config-only MPVSS reference is classified as:

```text
MPVSS config-only candidate: MAJOR ADAPTATION
selected frozen model: UNKNOWN because no model was selected
```

The Round 6 `StructuredReusePipeline` was not modified. Any future SegFormer
candidate must be statically audited for F1/F2/F3/F4 outputs, Stage-4
independence, decoder inputs, and final F4 cache shape before implementation.

## No Smoke or Benchmark

```text
checkpoint download: not completed
checkpoint SHA256: not applicable
model config snapshot: not applicable
forward smoke: not run
prediction sanity: not run
mIoU: not run
temporal reuse: not run
```
