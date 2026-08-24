# DriftGuard Round 7 Dataset Audit

## Audit Status

```text
dataset: VSPW
dataset access: BLOCKED
label-space status: INCOMPATIBLE for the frozen checkpoint
formal ground-truth benchmark: NOT RUN
```

This audit was performed before any Round 7 mIoU benchmark. No existing
small real-video set was substituted for VSPW.

## Source and Official Protocol

The official VSPW dataset-download repository is:

```text
VSPW-dataset/VSPW-dataset-download
```

Its README provides:

```text
full dataset: approximately 43 GB
480P package: separate official download
frame extraction: official scripts
```

The official README states that masks contain values from `0` through `124`
and `255`. Its evaluation rule treats both `0` and `255` as void:

```text
label[label == 0] = 255
label = label - 1
label[label == 254] = 255
```

Therefore the valid VSPW semantic class IDs after normalization are
`0..123`, with `255` as ignore/void.

The repository also provides `vspw_videolist.txt`, which contains the
official video identifiers and frame-count metadata. No random split or
custom video subset was created in this round.

## Local and Cloud Availability

Checked locations:

```text
local: D:/DriftGuard/data/raw/VSPW
cloud: /root/autodl-tmp/DriftGuard/data/raw/VSPW
```

Both locations were absent. The cloud machine has ample free storage, but
the official 480P Google Drive endpoint did not respond within the bounded
download probe. The official repository exposes Google Drive and Baidu
Pan links, but these require an external/manual download path in the current
environment. The dataset was therefore not silently replaced or partially
inventoried.

```text
DATASET ACCESS BLOCKED
```

## Frozen Checkpoint Audit

The frozen Round 6 checkpoint is the local SegFormer-B0 checkpoint loaded from:

```text
/root/autodl-tmp/models/segformer-b0
```

Its actual configuration reports:

```text
num_labels: 150
checkpoint label space: ADE20K-style labels
processor size: 512 x 512
resize: enabled
normalization: enabled
```

The checkpoint's `id2label` contains ADE20K labels such as `wall`, `building`,
`sky`, `floor`, `tree`, `road`, and 150 total categories.

VSPW's official mask protocol defines a separate 124-class semantic taxonomy
with `0` and `255` reserved as void under the official conversion. The
checkpoint therefore cannot be evaluated by direct integer-ID comparison.
No official, complete, one-to-one mapping from this ADE20K checkpoint label
space to VSPW's 124 classes was found in the project or in the official
download repository.

```text
LABEL SPACE STATUS: INCOMPATIBLE
```

This is a validity failure for the frozen Round 7 benchmark, not evidence
that temporal reuse damages VSPW accuracy.

## Dataset Fields Not Measurable

Because the official data package is unavailable, the following fields remain
unmeasured and are intentionally not fabricated:

```text
dataset version
official evaluated split
number of videos in split
number of evaluated frames
per-video frame counts in the installed package
image dimensions
annotation dimensions
frame naming in the installed package
sequence boundaries in the installed package
```

The repository metadata confirms that VSPW is a video/frame segmentation
dataset, but it does not replace an installed dataset manifest.

## Decision

Round 7 cannot proceed to Full mIoU, FI-2, per-class IoU, bootstrap, motion
analysis, or accuracy/latency Pareto without:

1. an accessible official VSPW package and official split;
2. a frozen checkpoint whose output label space is compatible with VSPW, or
   a documented, complete, reproducible mapping.

The proper next experiment is a dataset-compatible frozen segmentation model
benchmark. No fine-tuning, checkpoint swapping, category merging, or use of
the existing weakly labeled real-video set is authorized under this round.
