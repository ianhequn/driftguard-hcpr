# DriftGuard Round 4 Final Report

## Decision

```text
ROUND 4: STRONG ENGINEERING GO

DEPLOYMENT READINESS:
SELECTOR READY FOR PARTIAL-COMPUTE INTEGRATION
```

The frozen AFVT selector can be executed below the project's deployment
readiness line without changing its policy or scientific behavior.

At the representative 37.5% budget, the formal CUDA-resident selector mean is
`1.232 ms/frame` and P95 is `1.332 ms/frame`. All three core budgets have mean
below 2 ms and P95 below 3 ms. Policy equivalence is 100%.

This is selector implementation readiness only. Real Partial Computation was
not implemented or measured in Round 4.

## A. Source Semantics

The executable Round 3 visual-risk semantics are:

```text
reference RGB = immediately previous decoded video frame
residual = abs(current.float()/255 - previous.float()/255)
grid = 4x4 logical RGB regions
resize = PyTorch bilinear to 16x16, align_corners=False
metadata = cache age standardized by the frozen checkpoint
predictor = TinyRiskCNN Patch+Age, FP32, eval/inference mode
```

The reference is not a per-block historical RGB frame selected by cache age.
The Stage-4 feature cache remains per policy and budget, while the RGB risk
reference is shared previous-frame RGB. The Round 4 brief used the phrase
"cached reference frame", but source code and stored Round 3 outputs resolved
this terminology ambiguity before implementation.

Actual frozen videos are RGB uint8 at `480x640`. The optimized implementation
supports any H/W divisible by the 4x4 grid.

## B. Optimized Pipeline

The final API accepts CUDA-resident BCHW RGB tensors:

```python
selected_blocks = selector.select(
    current_frame_gpu,
    previous_frame_gpu,
    block_ages,
    refresh_ratio,
)
```

Execution order:

```text
CPU AFVT age partition over 16 values
-> return immediately when visual ranking is unnecessary
-> CUDA normalized RGB residual
-> vectorized 4x4 block partition
-> index AFVT tie candidates
-> batched CUDA bilinear resize
-> one TinyRiskCNN call
-> copy at most 16 risk scores to CPU
-> NumPy lexsort with lower-index deterministic tie break
```

No OpenCV/PIL resize, per-block CPU crop loop, full-image GPU-to-CPU round
trip, FP16, quantization, TensorRT, ONNX, Triton, or custom kernel is used.

## C. Strategy A versus Strategy B

Validation profiling selected Strategy A before the formal test profile:

| Strategy | 37.5% mean ms | Result |
|---|---:|---|
| A: vectorize all 16 blocks, then index candidates | 0.795 | Selected |
| B: direct candidate pixel gather | 0.847 | Rejected |

Direct gather copies fewer pixels but pays additional index-gather overhead.
The simpler all-block view/partition is faster and more stable for these small
4x4 workloads.

## D. Optimization Waterfall

Representative budget: 37.5%. Speedup uses the frozen Round 3 `38.404 ms`
baseline. Every stage has 100% selected-block equivalence.

| Stage | Mean ms | P95 ms | Speedup | Exact match |
|---|---:|---:|---:|---:|
| O0 original recheck | 48.437 | 98.074 | 0.79x | 100% |
| O1 GPU residual | 1.459 | 1.647 | 26.32x | 100% |
| O2 GPU patch | 1.103 | 1.525 | 34.83x | 100% |
| O3 batched GPU resize | 0.816 | 0.878 | 47.07x | 100% |
| O4 candidate batch | 0.768 | 0.821 | 50.01x | 100% |
| O5 fast path | 0.768 | 0.814 | 50.00x | 100% |
| O6 direct-gather trial | 0.811 | 0.864 | 47.38x | 100% |
| O7 final Strategy A | 0.820 | 1.270 | 46.81x | 100% |

The O0 recheck has large CPU preparation variance and is slower than the
frozen Round 3 result. Scientific speedup claims therefore use the frozen
`38.404 ms` source-of-truth baseline. The independent formal distribution
profile below is the source of truth for the engineering decision.

## E. Formal Latency

CUDA-resident input, FP32, warmup 200, measurements 1,000:

| Budget | Mean ms | Median ms | P95 ms | P99 ms | GPU event span ms |
|---:|---:|---:|---:|---:|---:|
| 25% | 0.126 | 0.078 | 0.772 | 0.847 | 0.107 |
| 37.5% | 1.232 | 1.234 | 1.332 | 1.567 | 1.205 |
| 50% | 0.114 | 0.094 | 0.116 | 0.953 | 0.093 |

The formal 37.5% selector implementation speedup is:

```text
38.404 / 1.232 = 31.17x
```

This is a selector implementation speedup, not model inference or complete
system speedup.

## F. Host-to-Device Boundary

When both RGB frames start as pageable CPU NumPy arrays, mean latency becomes:

| Budget | CUDA-resident mean ms | Including H2D mean ms |
|---:|---:|---:|
| 25% | 0.126 | 12.545 |
| 37.5% | 1.232 | 18.450 |
| 50% | 0.114 | 8.998 |

The deployment-ready decision therefore depends on the explicit Round 4 input
contract: current and reference RGB tensors are already CUDA resident. A
CPU-camera ingestion pipeline would require a separate pinned/asynchronous
upload design and cannot claim the CUDA-resident latency directly.

## G. Latency Breakdown

Independent 37.5% component microbenchmarks:

| Component | Mean ms | P95 ms |
|---|---:|---:|
| Age logic | 0.067 | 0.070 |
| Residual | 0.105 | 0.110 |
| Block extraction | 0.080 | 0.084 |
| Resize | 0.065 | 0.069 |
| Metadata | 0.064 | 0.067 |
| TinyRiskCNN | 0.453 | 0.478 |
| Ranking | 0.105 | 0.110 |
| Total in breakdown run | 0.790 | 0.817 |

Components were measured independently and are not expected to sum exactly to
the separately measured total. TinyRiskCNN is now the largest single
component, but the complete selector remains below the readiness line.

## H. Conditional Compute

Frozen Round 3 usage is preserved:

| Budget | CNN blocks/frame | Frames using CNN | Fast-path mean ms | Visual-path mean ms |
|---:|---:|---:|---:|---:|
| 25% | 0.766 | 6.38% | 0.097 | 1.205 |
| 37.5% | 6.298 | 100.00% | N/A | 1.048 |
| 50% | 0.340 | 2.13% | 0.079 | 0.813 |

The low 25% and 50% overall means come from the frozen AFVT fast path. At
37.5%, every evaluated frame requires visual tie breaking.

## I. Correctness

Formal frozen-test equivalence:

```text
videos: 3
frame-budget rows: 423
policy exact match: 423/423 = 100%
maximum risk-score difference: 1.0252e-5
mean risk-score difference: 2.8871e-7
```

Primary metric reproduction:

| Budget | Round 3 | Round 4 |
|---:|---:|---:|
| 25% | 0.017958915 | 0.017958917 |
| 37.5% | 0.012978836 | 0.012978835 |
| 50% | 0.010579057 | 0.010579057 |

Maximum stale run remains exactly:

```text
25% = 3
37.5% = 2
50% = 1
```

KL and Pixel Disagreement reproduce to numerical tolerance. The Round 3
`SECONDARY METRIC WARNING` remains active at 37.5% because AFVT Pixel
Disagreement is `0.008814`, about 5.6% worse than Age's `0.008350`.

## J. Memory

| Item | Value |
|---|---:|
| Peak additional allocated GPU memory | 14.063 MB |
| Frozen predictor parameters | about 0.022 MB FP32 |
| Persistent selector image buffer | none |
| Persistent direct-gather index map in final path | none |

The implementation remains well below the internal 100 MB additional-memory
target.

## K. Engineering Decision

All frozen STRONG ENGINEERING GO conditions pass:

```text
25%, 37.5%, 50% mean < 2 ms
25%, 37.5%, 50% P95 < 3 ms
policy equivalence = 100%
additional GPU memory < 100 MB
```

Therefore:

```text
ROUND 4 = STRONG ENGINEERING GO
DEPLOYMENT READINESS =
SELECTOR READY FOR PARTIAL-COMPUTE INTEGRATION
```

AFVT remains the scientifically locked scheduler. Round 4 does not reopen
scheduler research.

Authorized next research topic:

```text
Real Partial Computation + Accuracy / Latency Pareto
```

It is a separate phase and was not executed here.
