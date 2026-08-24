# DriftGuard Round 5 Final Report

## Decision

```text
ROUND 5 = REAL PARTIAL NO-GO

REAL PARTIAL COMPUTE ENGINEERING NO-GO
EARLY STOP B TRIGGERED
```

AFVT did not obtain measured end-to-end inference acceleration in Round 5.
The only authorized PyTorch-level approximate partial route was slower than
the complete dense Stage 4 at every core budget. Per the Round 5 hard stop,
model integration, task-quality evaluation, and a measured accuracy/latency
Pareto were not run.

No result in this report is described as real inference speedup.

## A. Compute Economics

Formal environment and boundary:

```text
Date: August 24, 2026
GPU: NVIDIA GeForce RTX 4090
PyTorch: 2.1.2+cu118
Transformers: 4.44.2
Model: frozen SegFormer-B0 ADE20K checkpoint
Input: CUDA-resident FP32 pixel_values, 1x3x512x512
Output: native segmentation logits ready
Warmup: 200
Measurements: 1000
Timing: CUDA Event
```

The manually decomposed path matched the Hugging Face forward exactly:

```text
max abs error: 0
mean abs error: 0
relative L2: 0
```

| Component | Mean ms | P95 ms | Runtime share vs direct full |
|---|---:|---:|---:|
| Stage 1 | 5.361 | 6.222 | 23.75% |
| Stage 2 | 5.457 | 6.312 | 24.18% |
| Stage 3 | 5.563 | 6.445 | 24.65% |
| Stage 4 | 4.612 | 5.558 | 20.43% |
| Decoder | 1.970 | 2.433 | 8.73% |
| Direct full model | 22.571 | 25.455 | 100% |

Independently timed components do not sum exactly to the separately measured
direct full forward because each interval has event/instrumentation effects.
The direct full model remains the formal system baseline.

The physically valid ideal equation is:

```text
T_ideal(r) = T_other + r * T_stage4 + T_selector(r)
```

| Budget | Ideal total ms | Ideal system speedup | Break-even partial Stage 4 ms |
|---:|---:|---:|---:|
| 25% | 19.238 | 1.173x | < 4.487 |
| 37.5% | 20.921 | 1.079x | < 3.380 |
| 50% | 20.380 | 1.108x | < 4.498 |

Early Stop A was not triggered. Cut 4 had theoretical but limited Amdahl
headroom, so the dependency audit and one partial route were justified.

## B. Stage 4 Dependencies

Actual B0 Stage 4:

```text
fresh F3 [B,160,32,32]
-> overlapping Conv2d 3x3 stride 2 patch embedding
-> 256 tokens at 16x16
-> two transformer blocks
-> final LayerNorm
-> F4 [B,256,16,16]
```

Each block contains token-wise LayerNorm, Q/K/V projections, global attention,
attention output projection, residuals, and MixFFN with a 3x3 depthwise
convolution. Stage 4 has `sr_ratio=1`; despite the class name
`EfficientSelfAttention`, no spatial reduction occurs in this stage.

A selected query still requires all 256 K/V context tokens. More importantly,
Block 2 current K/V depend on exact Block-1 outputs for all tokens. Exact final
selected outputs therefore require full patch embedding, full Block 1,
all-token Block-2 K/V, and selected/halo Block-2 work.

```text
EXACT PARTIAL: COMPUTATIONALLY POINTLESS
EXACT PARTIAL NO-GO

APPROXIMATE PARTIAL: FEASIBLE
```

The complete operator matrix and source-grounded reasoning are in
`reports/round5_stage4_dependency_audit.md`.

## C. Partial Route

The only implemented route was:

```text
Cached-Context Approximate Partial
```

Cache contents:

| Cache | Shape per layer | Dtype | Update rule |
|---|---|---|---|
| Patch tokens | [1,256,256] | FP32 | selected tokens |
| K | [1,8,256,32] | FP32 | selected tokens |
| V | [1,8,256,32] | FP32 | selected tokens |
| MixFFN dense1 | [1,256,1024] | FP32 | selected tokens |
| Block output | [1,256,256] | FP32 | selected tokens |
| Final F4 tokens | [1,256,256] | FP32 | selected tokens |

Selected tokens use current sparse patch projection, Q/K/V, attention, and
FFN. Non-refresh tokens do not execute the current full transformer path.
Their cached K/V provide stale global context, and cached dense1 values provide
stale 3x3 MixFFN neighbor context.

```text
logical refresh ratio = physical selected-token compute ratio
halo tokens computed = 0
neighbor context = cached stale dense1, therefore approximate
```

The engine exposes:

```python
stage4_output, cache_state, stats = engine.partial_stage4(
    stage3_feature,
    cache_state,
    refresh_mask,
)
```

## D. Correctness

The five required Round 5 test files were added. The complete cloud test suite
passed:

```text
55 passed
```

Specific Route-A checks:

```text
100% refresh max abs error: 0
100% refresh relative L2: 0
100% refresh allclose: PASS
selected cache entries update: PASS
unselected cache entries unchanged: PASS
2/4/6/8/12/16 block budgets: PASS
fresh F1-F3 + partial F4 + decoder integration contract: PASS
physical token counters: PASS
```

Formal profiling retained cache state across 1,200 warmup/measurement calls
per budget instead of resetting it every frame.

## E. Real Stage-4 Latency

This is an isolated Stage-4 hard-gate benchmark, not end-to-end inference.
Both full and partial paths use the same real B0 Stage-3 features and CUDA
Events with 200 warmup and 1,000 measurements.

| Method | Budget | Physical tokens | Mean ms | Median ms | P95 ms | Stage-4 speedup |
|---|---:|---:|---:|---:|---:|---:|
| Full Stage 4 | 100% | 256 | 4.366 | 4.170 | 5.107 | 1.000x |
| Cached-context partial | 12.5% | 32 | 6.780 | 6.540 | 7.987 | 0.644x |
| Cached-context partial | 25% | 64 | 6.741 | 6.499 | 7.955 | 0.648x |
| Cached-context partial | 37.5% | 96 | 6.662 | 6.491 | 7.581 | 0.655x |
| Cached-context partial | 50% | 128 | 6.914 | 6.468 | 9.901 | 0.631x |
| Cached-context partial | 75% | 192 | 6.781 | 6.588 | 7.619 | 0.644x |
| Cached-context full refresh | 100% | 256 | 4.572 | 5.509 | 6.063 | 0.955x |

All three core partial points violate:

```text
T_partial < T_fullStage4
```

The component-composed projection, including the frozen selector, is also
below 1.0x:

| Budget | Projected total ms | Projected speedup |
|---:|---:|---:|
| 25% | 24.825 | 0.909x |
| 37.5% | 25.853 | 0.873x |
| 50% | 24.987 | 0.903x |

These are projections from separately measured components, not measured
end-to-end results.

## F. Accuracy

```text
NOT RUN - EARLY STOP B
```

The Round 3 offline AFVT reference remains frozen:

| Budget | Logit Relative L2 | KL | Pixel Disagreement |
|---:|---:|---:|---:|
| 25% | 0.017959 | 0.004282 | 0.011275 |
| 37.5% | 0.012979 | 0.002393 | 0.008814 |
| 50% | 0.010579 | 0.001491 | 0.005988 |

The historical 37.5% warning remains active: AFVT Pixel Disagreement was about
5.6% worse than Age (`0.008814` versus `0.008350`).

No `Damage_implementation`, mIoU, or Real-AFVT quality claim is made because
the route failed its latency gate before frozen-video integration.

## G. Accuracy / Latency Pareto

No measured Pareto exists. Placeholder figures explicitly record the Early
Stop B reason rather than fabricating quality points.

The engineering plots that remain valid are:

```text
refresh_ratio_vs_real_latency.png
ideal_vs_measured_speedup.png
```

They show real isolated Stage-4 latency and a clearly labeled
component-composed system projection.

## H. Amdahl Analysis

Only Stage 4 was eligible for computation reduction. Its formal measured share
was 20.43%, so even ideal linear scaling produced only `1.079x-1.173x` at the
core budgets after selector cost.

The actual PyTorch route replaced efficient dense 16x16 transformer kernels
with many small operations:

```text
index selection
padding and neighborhood gathers
small matrix multiplications
cache scatter writes
per-layer K/V cache reads
Python/PyTorch dispatch overhead
```

At only 256 Stage-4 tokens, reduced arithmetic did not compensate for lost
dense-kernel efficiency and cache-management overhead. Sparse efficiency was
negative at all core points.

## I. Memory

| Item | Value |
|---|---:|
| Persistent per-layer and output cache | 4.0 MB |
| Peak temporary at 25% | 6.84 MB |
| Peak temporary at 37.5% | 9.62 MB |
| Peak temporary at 50% | 12.41 MB |
| Additional persistent memory warning | No |

Memory was not the blocker and remained far below the internal 1 GB warning
line.

## Final Answer

Round 5 asked:

```text
Does AFVT produce real, measurable end-to-end inference acceleration?
```

Answer:

```text
No, not with the current SegFormer-B0 Cut-4 design and the permitted
PyTorch-level partial-computation route.
```

Compute economics were initially sufficient to investigate, but exact partial
semantics were computationally pointless and cached-context sparse execution
was slower than dense Stage 4. Therefore:

```text
ROUND 5 = REAL PARTIAL NO-GO
```

The project stops here as required. No custom CUDA, Triton, TensorRT, model
replacement, dataset expansion, or later-round benchmark was started.
