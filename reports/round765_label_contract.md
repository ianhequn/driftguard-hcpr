# DriftGuard Round 7.65 Frozen Label Contract

## Canonical Contract

```text
raw input: official VSPW PNG mask
official normalization: 0 -> 255; subtract 1; 254 -> 255
evaluation-valid normalized IDs: 0..123 inclusive
canonical ignore index: 255
out-of-range normalized IDs: canonicalize to 255
raw 253: void / excluded, never remapped to a semantic class
loss: CrossEntropy(ignore_index=255)
metrics: confusion matrix over canonical labels 0..123; ignore 255
```

The project rule is generalized to all normalized IDs outside `0..123`, not a
special-case `if raw == 253`. This exactly matches the official evaluator's
validity predicate `0 <= normalized_gt < 124`.

## Training, Validation, Metrics

The same canonical label representation is used by all three paths:

```text
training loss       -> canonical 255 is ignored by CrossEntropy
validation loss     -> canonical 255 is ignored by CrossEntropy
validation metrics  -> canonical 255 is excluded from confusion/mIoU
```

The train split currently contains no raw253 pixels, but the behavior is now
defined for future data and for validation. Original PNG files remain
unchanged.

## Equivalence Gates

```text
synthetic raw [0, 1, 124, 253, 255] validity mask: PASS
synthetic confusion counts: PASS
17 real anomaly masks validity mask: PASS (17/17)
17 real anomaly masks confusion counts: PASS (17/17)
full train/val canonical scan: PASS
remaining invalid canonical labels: 0
CrossEntropy ignore semantics: PASS
mIoU void-pixel invariance: PASS
mIoU valid-pixel sensitivity: PASS
```

The implementation emits a runtime warning when previously out-of-range
normalized IDs are observed. The exhaustive audit records paths, counts,
dimensions, file sizes, unique IDs, and SHA256 rather than silently discarding
diagnostics.
