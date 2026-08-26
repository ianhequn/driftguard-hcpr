# Round 10 Cache Memory Audit

## Decision

**CACHE MEMORY AUDIT = PASS**

The previous quality JSON stores the implementation's allocated persistent
cache after refresh. The pipeline always stores F2, F3, and F4 in
`BoundaryCacheState`, so that value is identical across active boundaries
within a model. The logical cache is boundary-specific and is reported
separately in the corrected Round 10 tables.

## Tensor Accounting

All byte counts use the profile tensor shape and the recorded tensor
`element_size()` (4 bytes per element for these persistent outputs):
`numel() * element_size()`. The compute autocast is BF16, but the
post-normalization tensors retained by this implementation are FP32.

| Model | Boundary | F2 bytes | F3 bytes | F4 bytes | Logical cache | Allocated cache |
|---|---|---:|---:|---:|---:|---:|
| B0 | s4 | 1,048,576 | 655,360 | 262,144 | 0.250 MiB | 1.875 MiB |
| B0 | s3_s4 | 1,048,576 | 655,360 | 262,144 | 0.875 MiB | 1.875 MiB |
| B0 | s2_s4 | 1,048,576 | 655,360 | 262,144 | 1.875 MiB | 1.875 MiB |
| B1 | s4 | 2,097,152 | 1,310,720 | 524,288 | 0.500 MiB | 3.750 MiB |
| B1 | s3_s4 | 2,097,152 | 1,310,720 | 524,288 | 1.750 MiB | 3.750 MiB |
| B1 | s2_s4 | 2,097,152 | 1,310,720 | 524,288 | 3.750 MiB | 3.750 MiB |

## Semantics

- Logical cache: F4 for S4, F3+F4 for S3-S4, and F2+F3+F4 for S2-S4.
- Allocated cache: the actual persistent F2/F3/F4 tensors retained by the
  current implementation, including tensors unused by a shallower active
  boundary.
- No model-quality, latency, or dataset experiment was rerun for this audit.

The corrected main table exposes both quantities as
`logical_cache_*` and `allocated_cache_*`; mIoU, latency, speedup, CI, and
stage-call counts are unchanged.
