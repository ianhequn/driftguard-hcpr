# DriftGuard Round 10: Hardware-Aware Temporal Reuse Boundary Characterization

## 1. Decision

**ROUND 10 = STRONG BOUNDARY RESULT**

The study produces a reproducible boundary-dependent accuracy/latency frontier on
both frozen VSPW-compatible SegFormer models. S4-only reuse provides a distinct
low-damage operating point, while S2-S4 reuse provides the largest measured speed
at a substantially higher task-quality cost. The result is an empirical boundary
characterization, not a new reuse architecture.

## 2. Frozen Experimental Contract

- Hardware: NVIDIA GeForce RTX 4090 D, driver 595.71.05.
- Runtime: PyTorch 2.1.2+cu118, CUDA runtime 11.8, Transformers 4.44.2,
  BF16, batch size 1 for latency.
- Models: frozen SegFormer-B0 and SegFormer-B1 VSPW-124 checkpoints.
- Dataset: VSPW validation, 343 videos, 24,392 RGB/GT frame pairs, 124 classes,
  ignore label 255. Cache state resets at each video boundary.
- Stage 1 is always fresh for S2-S4; Stage 1-2 are fresh for S3-S4; Stage 1-3
  are fresh for S4. The decoder runs for every frame.
- Intervals: FI-2, FI-4, FI-8. The first frame of each video always refreshes.
- Latency: 200 warmup iterations, 1,000 measurements, five repetitions, with an
  immediately paired Full baseline for every configuration.

## 3. Correctness and Physical Avoidance

The 100% refresh gate passed for B0 and B1 at S4, S3-S4, and S2-S4. Refreshed
outputs are exactly equivalent to native Full SegFormer within the tested floating
point comparison: max absolute error 0, mean absolute error 0, relative L2 0.

The semantic suite passed **29 tests**. The full quality matrix and smoke sequence
completed without crashes. Stage-call counters verify that skipped stages are not
executed: for example, B0 S2-S4 FI-8 executes S1 on all 24,392 frames and S2-S4
on 3,181 frames, while the decoder still executes on all 24,392 frames.

## 4. Runtime Composition

| Model | Full profile (ms) | Stage 4 share | S3-S4 share | S2-S4 share |
|---|---:|---:|---:|---:|
| B0 | 8.487 | 18.42% | 41.45% | 64.12% |
| B1 | 8.614 | 18.54% | 42.29% | 64.81% |

These are component profiles from the same RTX 4090 D environment. They describe
the runtime opportunity, not a direct end-to-end speedup claim. The measured
speedup remains bounded by fresh prefix computation, decoder cost, cache control,
and GPU execution efficiency.

## 5. Accuracy/Latency Results

The complete machine-readable table is:
`reports/round10_boundary_quality_latency_table.csv`.

The quality-only exports are:
`results/R10_quality/full_results.csv` and
`results/R10_quality/full_results.json`.

The most informative operating points are:

| Model | Point | Boundary / interval | mIoU drop (pp) | Mean speedup | 95% CI |
|---|---|---|---:|---:|---:|
| B0 | lowest damage | S4 / FI-2 | 0.091 | 1.068x | [1.036, 1.100] |
| B0 | highest speed | S2-S4 / FI-8 | 1.835 | 2.396x | [2.292, 2.492] |
| B1 | lowest damage | S4 / FI-2 | 0.118 | 1.111x | [1.061, 1.200] |
| B1 | highest speed | S2-S4 / FI-8 | 2.715 | 2.282x | [2.086, 2.479] |

Full mIoU is 0.301291 for B0 and 0.371595 for B1. At the conservative S4/FI-2
point, mIoU is 0.300381 for B0 and 0.370413 for B1. The complete table also
contains mean, P95, P99, FPS, pixel accuracy, class accuracy, and call counts.

## 6. S4-Only Versus Vanilla S3-S4 Reuse

S3-S4 is treated only as a vanilla high-level feature-reuse baseline; no alignment,
fusion, or TFC-specific mechanism is added.

- **B0:** S4/FI-2 gives 1.068x with 0.091 pp mIoU drop, while S3-S4/FI-2 gives
  1.254x with 0.184 pp drop. S4-only is the distinct conservative point; S3-S4
  trades additional quality loss for additional speed.
- **B1:** S4/FI-2 gives 1.111x with 0.118 pp drop, while S3-S4/FI-2 gives
  1.332x with 0.283 pp drop. The same ordering persists at the larger model.

Thus S4-only reuse has a distinct practical operating point relative to vanilla
S3-S4 reuse for both models.

## 7. Scale Comparison

The optimal boundary is **stable across model scale** for the two selected criteria:
S4/FI-2 is the lowest-damage point, and S2-S4/FI-8 is the highest-speed point for
both B0 and B1. The absolute speed and damage differ, so this does not imply that
all models or hardware will share the same optimum.

## 8. Historical Fine-Grained Result

The Round 5 fine-grained spatial partial-computation result is **NOT COMPARABLE**
to this formal Round 10 result because it used a different experiment boundary and
historical RTX 4090/profiling context. It remains useful context: fine-grained
partial execution previously failed to realize dense Stage-4 savings, whereas
whole-stage reuse now removes complete stage invocations and produces measurable
system-level speedups.

## 9. Memory and Semantics

Only final stage features are logically required by the boundary, but the current
implementation retains F2, F3, and F4 in `BoundaryCacheState` after refresh. The
corrected table therefore reports both logical cache cost and actual allocated
persistent cost:

| Model | Boundary | Logical cache | Allocated cache |
|---|---|---:|---:|
| B0 | S4 | 0.25 MiB | 1.875 MiB |
| B0 | S3-S4 | 0.875 MiB | 1.875 MiB |
| B0 | S2-S4 | 1.875 MiB | 1.875 MiB |
| B1 | S4 | 0.50 MiB | 3.75 MiB |
| B1 | S3-S4 | 1.75 MiB | 3.75 MiB |
| B1 | S2-S4 | 3.75 MiB | 3.75 MiB |

No route exceeds the 1 GB internal warning boundary. No K/V cache, alignment,
custom kernel, sparse operator, or decoder modification is used. The detailed
audit is in `reports/round10_cache_memory_audit.md`.

## 10. Interpretation

The data show a systematic runtime/quality ordering that is interpretable from
profiled stage composition. Deeper boundaries expose more runtime opportunity and
produce larger speedups, but their quality penalty grows with temporal staleness.
S4-only reuse is therefore the quality-preserving boundary; S2-S4 is the speed
frontier. Runtime opportunity predicts the direction of the result, but not the
exact speedup, because the fresh prefix and decoder remain dense and the GPU-level
reuse path has nonzero control and cache overhead.

## 11. Limitations and Claim Boundary

This is a controlled VSPW validation characterization on one RTX 4090 D and two
frozen model scales. It is not a claim of universal optimality, a new temporal
reuse method, or general hardware portability. The latency study measures a
CUDA-resident input-to-logits path and excludes video decoding, disk I/O, pageable
H2D, visualization, and CSV writing.

The defensible paper framing after Round 10 is: **controlled, hardware-aware,
model-scale boundary characterization of measured accuracy/latency trade-offs in
hierarchical segmentation Transformer reuse**.

## 12. Artifacts

- Main table: `reports/round10_boundary_quality_latency_table.csv`
- Call table: `reports/round10_call_table.csv`
- Profile table: `reports/round10_profile_table.csv`
- Quality exports: `results/R10_quality/full_results.csv`, `results/R10_quality/full_results.json`
- Analysis rows: `results/R10_E06_analysis/round10_rows.json`
- Figures: `results/R10_E06_analysis/R10_F01_boundary_pareto.pdf`,
  `R10_F02_quality_refresh.pdf`, `R10_F03_runtime_opportunity_vs_speed.pdf`
- State: `reports/round10_reuse_boundary_state.json`

## 13. Final Stop

Round 10 is complete. No new datasets, model scales, boundaries, adaptive
selectors, lower-level kernels, or deployment backends are opened by this round.
