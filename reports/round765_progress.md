# DriftGuard Round 7.65 Progress

## Objective

Resolve the VSPW label-semantics contract without training, benchmarking, or changing dataset files.

## Completed

### 1. Official source audit

- The VSPW README documents raw values `0..124` and `255`, with `0` meaning other and `255` meaning void.
- The official loader applies `0 -> 255`, subtracts `1`, then `254 -> 255`.
- The official evaluator includes only normalized labels in `[0, 124)` in its confusion matrix.
- Therefore a raw `253` becomes normalized `252` and is excluded by the official evaluator.

### 2. Full-domain scan

- Train masks: `197253`; invalid raw labels before canonicalization: `0`.
- Validation masks: `24392`; invalid raw labels before canonicalization: `68314` pixels in `17` masks.
- Raw-253 pixels: `68314`; affected validation videos: `1/343`.
- After canonicalization, remaining invalid labels: `0`.
- Local and cloud hashes for all 17 affected masks match.

### 3. Contract equivalence

- Synthetic raw values and every real raw-253 anomaly mask reproduce the official valid mask.
- Confusion matrices are equal under official evaluation and project canonical evaluation.
- Specialized contract suite: `12 passed`.

## Selected contract

Contract B is selected:

```text
official normalization first
valid semantic IDs: 0..123
canonical ignore index: 255
all normalized IDs outside 0..123 and not already 255 -> 255
raw 253 -> normalized 252 -> canonical 255
```

The rule is generalized to all out-of-range normalized IDs. It is not a raw-253-only exception.

## Decision

```text
ROUND 7.65 = LABEL CONTRACT RESOLVED
DATA STATUS = READY WITH CANONICAL EVALUATION CONTRACT
ROUND 7.6 RESTART = AUTHORIZED, NOT EXECUTED
```

Historical `Round 7.6 = MODEL TRAINING NO-GO` remains unchanged. No training, temporal reuse, latency benchmark, or formal mIoU benchmark was executed in Round 7.65.
