# DriftGuard Round 7.5 VSPW Source Audit

## Decision

```text
official source verified: YES
package acquired: NO
dataset status: BLOCKED
```

This round inspected official/public VSPW source metadata and performed a
bounded recovery attempt. No third-party mirror, re-upload, or unknown archive
was accepted as a benchmark source.

## Official Sources

| Source | Maintainer | Role | Status |
|---|---|---|---|
| `VSPW-dataset/VSPW-dataset-download` | VSPW dataset organization | official package download metadata and extraction scripts | verified |
| `VSPW-dataset/VSPW_baseline` | VSPW dataset organization | official baseline code and dataset preparation reference | verified |
| `VSPW-dataset/VSPW_code` | VSPW dataset organization | official evaluation repository | verified |
| `sssdddwww2/CVPR2021_VSPW_Implement` | public reimplementation | secondary code reference only | not an official package source |

Repository heads audited on August 24, 2026:

```text
VSPW-dataset/VSPW-dataset-download:
dd0acea3836074dd282d55cc386477268a6a5ecf

VSPW-dataset/VSPW_baseline:
2dcd3fc04ca44d6ff2a3a9b9e4cd9da4ed5e6f30

VSPW-dataset/VSPW_code:
73b98ceb92cd9639da210da14f1bb0e02946f846

sssdddwww2/CVPR2021_VSPW_Implement:
3ead088804fe34d3b826b656e026d50562a3f7bd
```

## Package Information

The official download README identifies:

```text
full VSPW package: approximately 43 GB
separate VSPW 480P package: available
contents: extracted RGB frames and ground-truth masks
official frame extraction scripts: provided
checksums: not provided in the audited README
license/usage notice: no explicit license file found in the audited repositories
```

The preferred 480P package was selected because it is the smallest official
package suitable for this benchmark. It must still contain both RGB frames and
semantic masks before being considered complete.

## Expected Structure

The project expects a split/video/frame layout compatible with:

```text
VSPW/
  <split or video root>/
    <video_id>/
      origin/ or images/
      mask/ or labels/
```

The exact installed package tree remains unverified because the package was not
acquired. The official `vspw_videolist.txt` was inspected for video IDs and
frame metadata, but it does not substitute for an installed manifest.

## Annotation Semantics

The official README documents mask values `0..124` and `255`, with `0` and
`255` treated as void. The official normalization is:

```python
label[label == 0] = 255
label = label - 1
label[label == 254] = 255
```

This gives valid class IDs `0..123` and ignore index `255`. A real-mask audit
over at least 20 masks and 5 videos was not possible because no masks were
installed.

## Download Attempts

```text
target:
/root/autodl-tmp/DriftGuard/data/raw/VSPW

official 480P Google Drive:
bounded curl probe: no response
bounded wget probe: HTTPS connection timed out
attempt count: 2
```

The cloud machine has approximately 130 GB free on `/root/autodl-tmp`, so
storage is not the blocker. The blocker is official package access through the
available non-interactive network path.

The official Baidu Pan link is recorded in the source README, but the current
environment has no verified non-interactive Baidu download client/session.
Using it would require manual download or a separately authenticated client.

```text
MANUAL DOWNLOAD REQUIRED
DATASET ACCESS BLOCKED
```

## Required Manual Recovery

Download the official 480P package from the source README, then place it at:

```text
/root/autodl-tmp/DriftGuard/data/raw/VSPW
```

Afterward run the future Round 7 recovery audit to create the manifest,
inspect at least 20 masks across 5 videos, verify the official split, and
check missing-file completeness. Do not start mIoU or reuse benchmarking
before those checks pass.
