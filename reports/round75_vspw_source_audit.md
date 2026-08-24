# DriftGuard Round 7.5 VSPW Source Audit

## Decision

```text
official source verified: YES
official 480P package acquired: YES
official damaged-frame repair applied: YES
dataset status: READY
```

## Provenance

The package source is the official `VSPW-dataset/VSPW-dataset-download`
repository maintained by the VSPW dataset organization:

```text
repository: https://github.com/VSPW-dataset/VSPW-dataset-download
package: VSPW_480p.zip
resolution: 480P
package URL: https://drive.google.com/file/d/1rRujAmy3mzYqI0NjrdyVANDfrnFnBEp1/view
archive bytes: 14843646874
local SHA256: 3f350427005c01efab49fc951af21e542a473f586107d5eb841174af51d13f4f
official checksum: UNKNOWN
license file: UNKNOWN
access requirement: browser download was required for the main package
```

The package contains extracted RGB frames, semantic masks, official
`train.txt`, `val.txt`, and `test.txt` files, and the official class dictionary.

## Integrity And Repair

Archive-to-extraction inventory matched exactly before repair:

```text
archive files: 472068
missing extracted files: 0
size mismatches: 0
```

The official archive contains the issue documented by its maintainer for
video `988_f_Xvy_DCPt8`. Twelve RGB files were zero bytes. The official repair
archive was downloaded from:

```text
https://drive.google.com/file/d/1kuQksbhw_sgB5n6fKzts2G9KZn41-q_N/view
bytes: 1904640
local SHA256: 241f1686b49fc72b1fcd394e35138700053cb3779391dfd7e8703a0b13bac960
```

All 44 replacement JPEGs in that archive were decoded successfully at
`853x480` and applied. Zero-byte RGB files remaining: `0`.

## Installed Layout

```text
D:/datasets/VSPW_480p/VSPW_480p/
  data/<video_id>/origin/*.jpg
  data/<video_id>/mask/*.png
  train.txt
  val.txt
  test.txt
  data.txt
  label_num_dic_final.json

D:/DriftGuard/data/raw/VSPW
  -> directory junction to the installed root above
```

## Deterministic Inventory

```text
videos: 3536
RGB frames: 250412
semantic masks: 221645
train: 2806 videos, 197253 RGB-mask pairs
validation: 343 videos, 24392 RGB-mask pairs
test: 387 videos, 28767 RGB frames, no published masks
split overlap: 0
missing split directories: 0
unassigned video directories: 0
```

The official validation split is therefore the ground-truth benchmark split.
No custom split was created.

## Annotation Semantics

Twenty real masks across twenty videos were fully decoded. Observed raw labels
were a subset of `0..124` plus `255`; observed normalized labels were a subset
of `0..123` plus `255`. The audited normalization remains:

```python
label[label == 0] = 255
label = label - 1
label[label == 254] = 255
```

Artifacts:

```text
results/R75_E00_vspw_dataset/dataset_manifest.csv
results/R75_E00_vspw_dataset/split_manifest.csv
results/R75_E00_vspw_dataset/mask_semantics_audit.csv
results/R75_E00_vspw_dataset/summary.json
results/R75_E00_vspw_dataset/sha256.txt
```
