# DriftGuard Round 6 Final Report

## 1. Final Conclusion

```text
ROUND 6 = STRUCTURED REUSE STRONG GO
DYNAMIC STAGE POLICY = NO-GO
```

The answer to the Round 6 question is **yes**: AFVT-style temporal
redundancy can be converted into real, measured end-to-end acceleration when
the complete Stage 4 is either refreshed densely or reused from a cached final
Stage-4 feature.

The strongest measured point was `FixedInterval-8` at `1.218x` paired
end-to-end speedup. The recommended engineering point is `FixedInterval-2`:
`1.108x` speedup with `0.00789` mean LogitRelativeL2.

This is a whole-stage dense reuse result. It is not a block-sparse,
partial-attention, or custom-kernel result. Round 5's block-sparse NO-GO is
unchanged.

## 2. Frozen Experimental Boundary

```text
Model: SegFormer-B0
Input: 512 x 512
Hardware: NVIDIA GeForce RTX 4090
Stage 1-3: fresh every frame
Stage 4: complete dense refresh or complete cached reuse
Decoder: executed every frame
Cache cut: final Stage-4 output
Test set: frozen three-video test sequence
```

No model, dataset, split, resolution, decoder semantics, selector model, or
refresh-ratio definition was changed. No custom CUDA, Triton, TensorRT,
multi-stage cache, K/V cache, or sparse token execution was introduced.

## 3. Cache Semantics

`StructuredReusePipeline` stores only the final F4 tensor:

```text
shape: [1, 256, 16, 16]
dtype: FP32
persistent memory: 0.25 MB
```

On refresh, the full Stage 4 executes and replaces the cache. On reuse, no
Stage-4 module is called and the cached F4 is passed to the decoder together
with fresh F1-F3. The first frame always refreshes. Cache age resets to zero
on refresh and increments on reuse.

The Stage-4 forward counter confirms actual computation avoidance rather than
post-hoc masking. At 100% refresh, the reuse pipeline is exactly equivalent to
Full inference:

```text
max abs error: 0
mean abs error: 0
relative L2: 0
allclose: true
```

## 4. Compute Economics

The direct Round 6 component reprofile used 200 warmup iterations and 1000
CUDA-event measurements:

| Component | Mean |
|---|---:|
| Direct Full model | 20.133 ms |
| Isolated Stage 4 | 4.427 ms |

The resulting ideal Amdahl upper bound is approximately `1.280x`. This is an
upper bound only; it assumes Stage 4 can be removed without policy, cache, or
measurement overhead. The formal system benchmark used a paired Full
reference of `21.502 ms`, and that paired value is used for all reported
speedups.

The measured ceiling is therefore consistent with the economics: removing
Stage 4 can produce a meaningful but bounded system gain, not a 2x model
speedup.

## 5. Authoritative Latency Protocol

For every policy:

```text
200 warmup iterations
1000 measurements
same frozen input sequence
same CUDA-resident input boundary
paired Full baseline immediately after the policy
same process
```

The measured boundary is CUDA-resident `pixel_values` to native segmentation
logits. It includes fresh Stage 1-3, policy execution, whole Stage-4 refresh
or reuse, and decoder execution. It excludes disk I/O, video decoding, CPU
preprocessing, pageable CPU H2D, logging, plotting, and CSV writing.

An earlier latency run showed impossible order-dependent speedups and was
rejected. The paired protocol above is the source of truth.

## 6. Accuracy / Latency Results

| Method | Refresh | Mean ms | P95 ms | FPS | Speedup | Logit L2 | KL | Pixel Disagreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 100.0% | 22.157 | 25.706 | 45.13 | 1.000x | 0 | 0 | 0 |
| FixedInterval-2 | 50.0% | 19.223 | 21.991 | 52.02 | 1.108x | 0.00789 | 0.001667 | 0.006061 |
| FixedInterval-3 | 33.3% | 18.508 | 21.819 | 54.03 | 1.161x | 0.01230 | 0.003120 | 0.009464 |
| FixedInterval-4 | 25.0% | 18.745 | 24.503 | 53.35 | 1.138x | 0.01630 | 0.004950 | 0.012699 |
| FixedInterval-6 | 16.7% | 18.102 | 22.800 | 55.24 | 1.180x | 0.02098 | 0.007950 | 0.016264 |
| FixedInterval-8 | 12.5% | 17.755 | 21.522 | 56.32 | 1.218x | 0.02439 | 0.010738 | 0.019168 |
| BestVisualRisk | 64.6% | 22.613 | 26.790 | 44.22 | 0.950x | 0.00449 | 0.000821 | 0.004341 |
| BestStageAgeVisual | 64.6% | 22.688 | 27.691 | 44.08 | 0.963x | 0.00449 | 0.000821 | 0.004341 |

`FixedInterval-2` and `BestStageAge` have equivalent refresh behavior and
quality in this frozen sequence. The fixed schedule is preferable because it
has no selector cost.

## 7. Dynamic Policy Decision

The frozen validation selection used:

```text
aggregation: max
visual threshold: -2.2600908279418945
StageAgeVisual max_age: 5
validation Spearman: approximately 0.710
```

The predictor signal was not rejected by the validation correlation rule.
However, deployment latency is the deciding factor. Visual-risk policies
added approximately `2.13-2.14 ms` selector latency and measured below Full:

```text
BestVisualRisk: 0.950x
BestStageAgeVisual: 0.963x
```

Therefore dynamic stage selection is a **NO-GO for the current frozen
pipeline**. This does not invalidate the risk predictor as an offline
analysis signal; it means its runtime cost is larger than the whole-stage
reuse benefit at this cut.

## 8. Correctness and Test Evidence

Round 6 added and ran the whole-stage reuse test suite:

```text
tests/test_stage4_whole_cache.py
tests/test_stage4_reuse_skip.py
tests/test_stage4_refresh_state.py
tests/test_stage4_fixed_interval.py
tests/test_stage4_dynamic_policy.py
tests/test_structured_reuse_full_refresh.py
```

Cloud validation result:

```text
20 passed
```

The tests cover first-frame refresh, refresh/reuse state transitions, cache
age, fixed and dynamic policy decisions, Stage-4 call avoidance, cache
stability for unselected frames, valid refresh rates, and 100% refresh
equivalence.

## 9. Memory and Engineering Cost

```text
Persistent final-F4 cache: 0.25 MB
Reuse benchmark peak temporary memory: 276.125 MB
Visual selector mean: approximately 2.13-2.14 ms
```

The route does not introduce a large persistent cache. Its main engineering
cost is the quality trade-off caused by holding F4 stale, plus the selector
cost for dynamic policies.

## 10. Scientific Interpretation

The result supports a narrow claim:

> On the frozen SegFormer-B0 CUDA-resident pipeline and frozen three-video
> test sequence, complete Stage-4 temporal reuse produces measurable
> end-to-end acceleration, with a quality/latency trade-off controlled by
> refresh interval.

It does not establish generalization to VSPW, UAVid, other SegFormer sizes,
other hardware, or multi-stage reuse. The result also does not show that a
visual-risk selector is beneficial in the deployed runtime.

The historical Round 3 warning remains relevant: at approximately 37.5%
refresh, AFVT pixel disagreement had previously exceeded Age by about `5.6%`.
Round 6 does not erase that warning; it evaluates a different whole-stage
reuse route and keeps the quality metrics explicit.

## 11. Final Decision and Stop Boundary

```text
ROUND 6 = STRUCTURED REUSE STRONG GO
```

The next action is to stop this round and preserve the artifacts. Do not
automatically start multi-stage reuse, new datasets, SegFormer-B1/B2,
custom CUDA/Triton/TensorRT work, Jetson benchmarking, or paper writing.
