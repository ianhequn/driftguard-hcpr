# DriftGuard Round 7 Final Report

## Final Decision

```text
ROUND 7:
MODEL-DATASET COMPATIBILITY NO-GO

PRIMARY POLICY:
FixedInterval-2 (pre-frozen, not evaluated because the validity gate failed)

GROUND-TRUTH VALIDATED SPEEDUP:
NO
```

Round 7 did not produce a ground-truth mIoU result. This is intentional:
the formal benchmark was stopped at the dataset/model validity gate rather
than reporting an invalid number.

## Q1. Is the Dataset Available?

No. The official VSPW package is not present at either:

```text
D:/DriftGuard/data/raw/VSPW
/root/autodl-tmp/DriftGuard/data/raw/VSPW
```

The official VSPW download repository was reachable and its metadata and
README were inspected. It provides a full package of approximately 43 GB and
a separate official 480P package. The bounded cloud probe of the official
480P Google Drive endpoint did not respond, and no completed package was
available through the current environment.

```text
DATASET ACCESS BLOCKED
```

The existing three-video Round 6 set was not promoted to a formal
ground-truth benchmark because it has no verified semantic segmentation GT.

## Q2. What Is the Official VSPW Annotation Semantics?

The official VSPW README states that mask values are `0..124` and `255`.
It defines `0` as "others" and `255` as void. The prescribed normalization is:

```text
label[label == 0] = 255
label = label - 1
label[label == 254] = 255
```

After normalization, VSPW has valid class IDs `0..123` and ignore index
`255`. This rule would be implemented before mIoU if the dataset were
available.

## Q3. Is the Frozen Checkpoint Label Space Compatible?

No direct compatibility was established.

The frozen Round 6 checkpoint is SegFormer-B0 with:

```text
num_labels: 150
id2label: ADE20K-style 150-category labels
processor: 512 x 512 resize and normalization
```

VSPW uses a separate 124-class taxonomy after void normalization. Direct
comparison of the checkpoint's 0..149 predictions with VSPW's 0..123
annotations would be invalid. No official complete reproducible mapping from
the frozen ADE20K label space to VSPW was found in the project or official
download repository.

```text
LABEL SPACE STATUS: INCOMPATIBLE
```

This is a model/dataset validity failure, not a temporal-reuse quality
failure.

## Q4-Q10. Why Are mIoU and Pareto Results Absent?

The required execution order is:

```text
dataset audit
-> label-space audit
-> Full mIoU
-> FI-2
-> remaining intervals
```

The first two gates failed. Therefore the following were not run:

```text
Full mIoU
FI-2/FI-3/FI-4/FI-6/FI-8 mIoU
per-class IoU
per-video metrics
bootstrap confidence intervals
motion-conditioned mIoU analysis
surrogate-vs-mIoU correlations
groundtruth_accuracy_latency_pareto.png
```

No surrogate metric was presented as a replacement for ground-truth mIoU.
Round 6's measured latency and LogitRelativeL2 results remain valid only
within the Round 6 frozen non-GT boundary.

## Scientific Interpretation

Round 7 currently answers:

> The existing Round 6 evidence cannot yet be promoted to a
> ground-truth-validated semantic segmentation claim.

It does not answer that whole-Stage-4 reuse fails on VSPW. It answers that the
requested frozen checkpoint/data pair is not currently a valid benchmark
configuration.

## Required Next Experiment

Only this follow-up is justified:

```text
dataset-compatible frozen segmentation model benchmark
```

That requires an official VSPW package and a frozen checkpoint trained for the
VSPW label space, or a documented complete mapping that passes a separate
validity audit. Do not fine-tune, merge categories to improve mIoU, reuse the
unlabeled Round 6 videos as GT, or change the temporal reuse semantics.

Round 7 stops here.
