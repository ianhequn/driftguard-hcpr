# DriftGuard Round 7.65 Official VSPW Semantics Audit

## Scope

This round audits the label/evaluation protocol only. No model training,
temporal reuse, formal mIoU benchmark, latency measurement, or dataset-file
modification was performed.

## Source Evidence

| Source | Raw-domain rule | Normalization | Evaluation valid rule | Ignore behavior |
|---|---|---|---|---|
| Official download README | States raw values `0..124` and `255`; `0` is “others” and `255` is “void” | `0 -> 255`, subtract `1`, then `254 -> 255` | README gives the recommended normalization but does not spell out a matrix predicate | `0` and `255` are void after the recommended transform |
| Official `dataset.py` | Does not validate raw IDs before transforming them | Lines 95-102 execute `segm[segm==0]=255`, `segm=segm-1`, `segm[segm==254]=255` | Loader returns the transformed array; no range check here | `255` remains the transformed void value; an unexpected raw `253` becomes `252` |
| Official `utils.py` evaluator | Operates on normalized GT | Receives the loader output | `_generate_matrix` uses `(gt_image >= 0) & (gt_image < num_class)` before `bincount` | Any normalized value outside `[0,123]` is excluded from confusion counts |
| Released VSPW 480P package | Full scan found raw `253` in 17 validation masks | Official normalization maps it to `252` | Official evaluator excludes the resulting `252` pixels by its range predicate | Project canonicalizes those excluded pixels to `255` for loss/metric interoperability |

Primary sources:

```text
https://github.com/VSPW-dataset/VSPW-dataset-download
https://github.com/VSPW-dataset/VSPW_baseline/blob/master/dataset.py
https://github.com/VSPW-dataset/VSPW_baseline/blob/master/utils.py
```

## Conflict

The README's documented raw domain does not include `253`, and the loader does
not reject it. The evaluator nevertheless has a general normalized valid-range
filter. The released package contains a small set of structurally valid PNGs
that exercise this discrepancy. The audit does not infer that `253` represents
any semantic class.

## Decision

Contract B is adopted because it reproduces the actual official evaluator:

```python
official_normalized = official_loader_normalization(raw)
official_valid = (official_normalized >= 0) & (official_normalized < 124)
canonical_gt = official_normalized.copy()
canonical_gt[~official_valid] = 255
```

The canonical project representation is therefore always a 124-class tensor
with IDs `0..123` plus ignore `255`. This is a representation of the official
valid-mask behavior, not a claim that raw `253` has a known semantic meaning.

## Artifact Classification

Raw `253` is classified as an **official released annotation artifact**, not a
local/cloud transfer corruption:

- all 17 files decode as valid PNGs at `853x480`;
- all contain the same ordinary semantic IDs plus `253`;
- the local and cloud copies have identical SHA256 for all 17 files;
- both installations derive from the same official VSPW 480P archive;
- the archive itself is not edited in this round.

This classification does not assign a semantic class to `253`; the final
semantic treatment is void/excluded under the official evaluator contract.
