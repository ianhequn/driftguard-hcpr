# DriftGuard Round 8 Final Report

日期：2026-08-25

## Final decision

```text
ROUND 8 = SOURCE CHECKPOINT BLOCKED
```

## Serial context

Round 7.1 passed its technical gate:

```text
ROUND 7.1 = ROBUST GO
FI-2 mean paired speedup = 1.0924x
FI-2 95% CI = [1.0337, 1.1510]
IMPLEMENTATION = VALID
```

The GPU was released before Round 8 source acquisition. This satisfies the serial resource boundary.

## Blocking condition

The pre-declared source `nvidia/segformer-b1-finetuned-ade-512-512` could not be obtained. Cloud Hugging Face access timed out at `huggingface.co:443`; local access timed out as well. No local B1 snapshot was available.

## Work not performed

The following were correctly not started:

- B1 architecture/initialization construction;
- 100-step smoke and batch calibration;
- 500-step throughput estimate;
- 40,000-step training;
- VSPW-124 B1 freeze and reload;
- B1 100% refresh equivalence;
- B1 Full/FI-2/FI-4/FI-8 GT benchmark;
- B1 latency, profile and cross-scale figures.

Using an unverified checkpoint, substituting B2/B3, or continuing with a partial B1 would violate the experiment contract. Therefore this is a valid source-acquisition stop, not a NO-GO for structured reuse on B1.

## Current scientific boundary

The valid results remain:

1. B0 whole-Stage4 reuse has a real robust FI-2 speedup on the frozen VSPW validation benchmark.
2. B1 model-scale generalization is **not evaluated**.

The task stops here as required.
