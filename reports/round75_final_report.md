# DriftGuard Round 7.5 Final Report

## Final Decision

```text
ROUND 7.5:
COMPATIBILITY RECOVERY BLOCKED

DATASET STATUS:
BLOCKED

MODEL STATUS:
NOT FOUND

SELECTED MODEL:
NONE

LABEL SPACE:
UNVERIFIED

STRUCTURED REUSE COMPATIBILITY:
UNKNOWN

BENCHMARK READINESS:
NOT READY
```

Round 7.5 did not run mIoU, Full inference, FI-2, temporal reuse, latency,
or any accuracy/latency benchmark.

## Q1. Is VSPW Installed?

No. Both checked roots are absent:

```text
D:/DriftGuard/data/raw/VSPW
/root/autodl-tmp/DriftGuard/data/raw/VSPW
```

The official VSPW download repository was audited. It identifies a roughly
43 GB full package and a separate official 480P package containing extracted
frames and masks. The preferred 480P Google Drive endpoint was probed with
bounded `curl` and `wget` attempts; the cloud connection timed out. The
official Baidu route requires a manual/authenticated download path that is
not available to the current non-interactive process.

```text
MANUAL DOWNLOAD REQUIRED
DATASET STATUS = BLOCKED
```

## Q2-Q4. Dataset Package, Split, and Real Mask Verification

```text
official package: not acquired
official split: not verified from installed data
real mask audit: not run
deterministic manifest: not generated because no files exist
```

The official README semantics were recorded, but the required audit of at
least 20 real masks across 5 videos could not be performed.

## Q5-Q8. Compatible Checkpoint Discovery

No valid VSPW-trained SegFormer checkpoint was found.

Audited sources:

```text
VSPW-dataset/VSPW_baseline
VSPW-dataset/VSPW_code
sssdddwww2/CVPR2021_VSPW_Implement
ziplab/MPVSS
```

The official baseline exposes PSPNet/OCRNet/CNN configurations and references
ImageNet-pretrained weights. The official code repository provides evaluation
code and directs users to contact the author for training/testing code. The
public reimplementation contains model source/configuration files but no
verified VSPW segmentation checkpoint artifact.

The official-author MPVSS repository contains a VSPW configuration with
`NUM_CLASSES: 124`, `IGNORE_VALUE: 255`, and official VSPW train/validation
dataset names. It is a configuration-only candidate: no VSPW checkpoint was
found, and its Mask2Former/MPVSS architecture would require major adaptation
for Round 6's Stage-4 cache contract.

The current ADE20K SegFormer-B0 is not a candidate:

```text
150 output classes
ADE20K-style label taxonomy
no verified VSPW class order
```

It was not truncated, remapped, fine-tuned, or used in this round.

## Q9-Q11. Class Order and Structured Reuse

```text
class alignment: 0/124 verified for a frozen checkpoint
checkpoint forward smoke: NOT RUN
architecture contract: available only as config-level MPVSS reference
structured reuse compatibility:
    MPVSS config-only candidate: MAJOR ADAPTATION
    selected frozen model: UNKNOWN
```

No public candidate reached the point where F1-F4 structure or decoder
compatibility could be audited. Round 6 `StructuredReusePipeline` remains
unchanged.

## Q12. Is Training Required?

Possibly, but only as a future fallback:

```text
if official VSPW becomes available and no compatible public checkpoint exists:
    train and freeze a VSPW-compatible SegFormer
```

Round 7.5 does not train or fine-tune. The current ADE20K checkpoint is not
fine-tuned.

## Q13. Can Round 7 Restart?

No.

The restart gate requires both:

```text
official VSPW package READY
VSPW-compatible frozen checkpoint READY
```

Neither condition is currently satisfied. The correct next task is manual
official data recovery followed, if needed, by a separate training round.

## Required Manual Action

Use the official VSPW download README to obtain the 480P package, then place
it at:

```text
/root/autodl-tmp/DriftGuard/data/raw/VSPW
```

Do not commit the dataset. After placement, rerun only the Round 7.5 dataset
inventory and mask/split audit. Do not start Round 7 restart automatically.
