# DriftGuard Round 9 Final Report

## Final Decision

```text
ROUND 9 = DATASET BLOCKED
BLOCK REASON = STORAGE CAPACITY
```

## Required Output

```text
DATASET: Cityscapes
VALIDATION TARGETS: unavailable
TEMPORAL SEQUENCE COVERAGE: not evaluated
B0: not run
B1: not run
PHASE AUDIT: not run
100% REFRESH EQUIVALENCE: B0 not run / B1 not run
CROSS-DATASET RESULT: NOT SUPPORTED
PUBLICATION EVIDENCE: UNCHANGED
```

## Reason

The logged-in Cityscapes official download page is accessible, but the data
has not been downloaded. The required sequence package is listed as 324GB;
with `gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip`, the
compressed requirement is about 335GB before extraction. The local D: drive
has about 245GB free and the cloud volume has about 95GB free. Without the
sequence files, the 500 labeled validation targets and their chronological
context cannot be reconstructed.

This is a data availability stop, not a temporal-reuse failure. Round 7.1 and
Round 8 results remain frozen and unchanged. No model, algorithm, timing
protocol, or dataset substitution was introduced.

Round 9 stops here. The next allowed action is to provide a storage location
with enough capacity for the official packages and extraction workspace, then
restart only from the dataset audit gate. No large download was started.
