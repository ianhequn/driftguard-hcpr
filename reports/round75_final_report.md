# DriftGuard Round 7.5 Final Report

## Final Decision

```text
ROUND 7.5:
DATA READY - MODEL TRAINING REQUIRED

DATASET STATUS:
READY

MODEL STATUS:
TRAINING REQUIRED

SELECTED MODEL:
NONE

LABEL SPACE:
DATASET VERIFIED / MODEL ALIGNMENT UNVERIFIED

STRUCTURED REUSE COMPATIBILITY:
UNKNOWN

BENCHMARK READINESS:
NOT READY
```

Round 7.5 did not run mIoU, Full inference, FI-2, temporal reuse, latency, or
any accuracy/latency benchmark.

## Dataset Recovery

The official VSPW 480P package is installed locally at
`D:/datasets/VSPW_480p/VSPW_480p` and exposed to the project through
`D:/DriftGuard/data/raw/VSPW`.

```text
package bytes: 14843646874
local SHA256: 3f350427005c01efab49fc951af21e542a473f586107d5eb841174af51d13f4f
videos: 3536
RGB frames: 250412
masks: 221645
manifest rows: 250412
```

The package's official damaged-frame issue was real: twelve files in
`988_f_Xvy_DCPt8/origin` had zero bytes. The official 1.90 MB repair archive
contained 44 replacement JPEGs; all decoded successfully at `853x480`, were
applied, and removed every zero-byte file.

## Split And Labels

The installed official split files are mutually disjoint and cover every
video directory:

| Split | Videos | RGB frames | Masks | Paired frames |
|---|---:|---:|---:|---:|
| train | 2806 | 197253 | 197253 | 197253 |
| validation | 343 | 24392 | 24392 | 24392 |
| test | 387 | 28767 | 0 | 0 |

The official test split has no masks in the released 480P package, so future
ground-truth evaluation must use the official validation split.

Twenty masks across twenty videos were fully decoded. Their raw values obeyed
the official `0..124, 255` domain; after official normalization they obeyed
`0..123, 255`, with `255` as ignore. Dataset taxonomy is verified.

## Model Discovery

No public checkpoint passed all required gates:

```text
checkpoint artifact exists
explicitly trained on VSPW
124 output classes
exact official VSPW class order documented
loadable architecture and forward smoke
```

The official VSPW repositories provide baseline/evaluation code but no verified
VSPW-trained SegFormer weights. The MPVSS repository supplies a VSPW 124-class
configuration but no checkpoint artifact and is not a direct Round 6
architecture match. The frozen ADE20K SegFormer-B0 remains invalid because it
outputs 150 ADE20K classes.

Consequently:

```text
valid frozen candidates: 0
checkpoint forward smoke: NOT RUN
class alignment: UNVERIFIED FOR ANY MODEL
selected model: NONE
structured reuse compatibility: UNKNOWN
```

## Readiness

The data half of the benchmark pair is now reproducible and ready. The model
half is not. Round 7 cannot restart until a VSPW-compatible segmentation model
is trained, frozen, and separately audited for exact class order and forward
correctness.

The only permitted next task is:

```text
Round 7.6:
Train and Freeze a VSPW-Compatible SegFormer
```

Training was not started in Round 7.5.
