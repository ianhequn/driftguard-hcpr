# DriftGuard Round 6 Whole-Stage Reuse Audit

## Frozen Boundary

```text
Round 5 block-sparse route: frozen NO-GO
Stage 1-3: always fresh
Stage 4: complete dense refresh or complete cached reuse
Decoder: always executed
```

Round 6 does not use partial tokens, partial attention, K/V caches, MixFFN
caches, gather/scatter token updates, or custom kernels.

## Cache Contract

Persistent model cache:

```text
cached_f4: final Stage-4 feature only
age: number of consecutive reuse frames after the last refresh
initialized: prevents first-frame reuse
```

Refresh:

```text
run complete dense Stage 4
replace cached_f4
age = 0
stage4_forward_calls += 1
```

Reuse:

```text
do not call any Stage-4 module
return cached_f4
age += 1
stage4_forward_calls unchanged
```

The decoder still consumes fresh F1/F2/F3 and the selected current-or-cached
F4 every frame.

## Execution Evidence

The cache route was validated with module-level Stage-4 forward counters:

```text
refresh: complete dense Stage 4 executes, counter increments
reuse: no Stage-4 module executes, counter is unchanged
first frame: refresh is forced
100% refresh: max abs error = 0, mean abs error = 0,
              relative L2 = 0, allclose = true
```

The persistent cache is one final F4 tensor with shape `[1, 256, 16, 16]`
and FP32 storage. Its measured size is `0.25 MB`. No per-layer hidden-state,
K/V, FFN-intermediate, or sparse-token cache is used.

## Measured Boundary

The authoritative latency boundary is CUDA-resident `pixel_values` input to
native segmentation logits. It includes fresh Stage 1-3, the refresh policy,
whole Stage-4 refresh or reuse, and the decoder. It excludes disk I/O, video
decode, CPU preprocessing, pageable H2D, logging, plotting, and CSV writing.

Each policy uses the same frozen test sequence, 200 warmup iterations, and
1000 measurements. A paired Full baseline is run immediately in the same
process and with the same input boundary. This paired result is the source of
truth for speedup.

The direct component reprofile measured Stage 4 at `4.427 ms` and direct Full
at `20.133 ms`, giving an Amdahl upper bound of approximately `1.280x` if
Stage 4 could be removed without overhead. The formal paired end-to-end Full
reference was `21.502 ms`; these two measurements are kept separate because
they serve different profiling purposes.

## Policy Decision

Fixed interval reuse produces real system-level acceleration:

```text
FixedInterval-2: 1.108x, LogitRelativeL2 0.00789
FixedInterval-3: 1.161x, LogitRelativeL2 0.01230
FixedInterval-8: 1.218x, LogitRelativeL2 0.02439
```

`FixedInterval-2` is the recommended practical point because it stays below
the internal `0.01` logit-damage filter while still exceeding `10%`
end-to-end speedup.

The visual-risk and StageAge+Visual policies are not deployment winners in
this round. Their frozen validation correlation was acceptable
(Spearman approximately `0.710` for max aggregation), but their selector
cost was approximately `2.13-2.14 ms`, resulting in paired speedups of
`0.950x` and `0.963x`. StageAge is equivalent to FixedInterval-2 and adds no
quality advantage.

## Boundary Conclusion

```text
WHOLE-STAGE DENSE REUSE: GO
DYNAMIC VISUAL STAGE POLICY: NO-GO
BLOCK-SPARSE PARTIAL COMPUTE: FROZEN NO-GO FROM ROUND 5
```
