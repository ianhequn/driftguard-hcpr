# DriftGuard Round 5 Compute Economics

## Measurement Contract

```text
GPU: NVIDIA GeForce RTX 4090
PyTorch: 2.1.2+cu118
Transformers: 4.44.2
Model: frozen SegFormer-B0 ADE20K checkpoint
Input: CUDA-resident FP32 pixel_values, 1x3x512x512
Output boundary: native segmentation logits ready
Warmup: 200
Measurements: 1000
Timing: CUDA Event
```

CPU `AutoImageProcessor` preprocessing is measured separately and excluded
from the CUDA-resident model-input-to-logits contract. Disk I/O, video decode,
pageable H2D, visualization, and CSV writing are excluded.

The manually decomposed forward exactly reproduced the Hugging Face model
forward (`max abs error = 0`, `relative L2 = 0`).

## Stage Breakdown

| Component | Mean ms | Median ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|
| Stage 1 | 5.361 | 5.194 | 6.222 | 7.715 |
| Stage 2 | 5.457 | 5.265 | 6.312 | 8.476 |
| Stage 3 | 5.563 | 5.365 | 6.445 | 8.742 |
| Stage 4 | 4.612 | 4.443 | 5.558 | 7.410 |
| Decoder | 1.970 | 1.883 | 2.433 | 3.307 |
| Direct full model | 22.571 | 22.136 | 25.455 | 28.902 |

The independently event-timed components sum to `22.963 ms`; the direct full
model is the formal system baseline at `22.571 ms`.

## Stage 4 Share

```text
T_full = 22.571 ms
T_stage4 = 4.612 ms
T_other = 17.959 ms
T_stage4 / T_full = 20.43%
```

Stage 4 is materially larger than the old informal estimate of about `2.6 ms`,
but it is still only one fifth of the complete model. Cut 4 is therefore
strictly Amdahl-limited.

## Ideal Linear Scaling

Component latencies add, so the physically valid equation is:

```text
T_ideal(r) = T_other + r * T_stage4 + T_selector(r)
```

The minus signs in the task brief are treated as a typographical error.

| Budget | Selector ms | Ideal partial Stage 4 ms | Ideal total ms | Ideal speedup |
|---:|---:|---:|---:|---:|
| 25% | 0.126 | 1.153 | 19.238 | 1.173x |
| 37.5% | 1.232 | 1.730 | 20.921 | 1.079x |
| 50% | 0.114 | 2.306 | 20.380 | 1.108x |

## Break-Even

For any net system speedup:

```text
T_selector(r) + T_partial(r) < T_stage4
```

| Budget | Maximum partial Stage 4 ms |
|---:|---:|
| 25% | 4.487 |
| 37.5% | 3.380 |
| 50% | 4.498 |

The 37.5% point has the tightest break-even boundary because the frozen AFVT
visual path costs `1.232 ms`.

## Early Stop A

```text
core budgets with ideal speedup < 1.03: 0 / 3
EARLY STOP A: NOT TRIGGERED
```

Cut 4 has enough theoretical headroom to justify the mandatory dependency
audit. This does not establish real speedup; it only authorizes one bounded
PyTorch-level partial-compute experiment.
