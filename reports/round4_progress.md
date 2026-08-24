# DriftGuard Round 4 Progress

## 2026-08-24 - Source Semantics Audit

- Optimization stage: Phase 4.0 / pipeline audit.
- Code change: none; Round 3 source and frozen artifacts inspected.
- Correctness result: frozen AFVT semantics are reconstructable.
- Mean latency: Round 3 baseline is 38.404 ms/frame at 37.5%.
- P95 latency: Round 3 baseline is 94.550 ms/frame at 37.5%.
- Speedup: 1.0x baseline.
- Next action: implement the GPU-resident FP32 selector without changing the
  scheduler, predictor, reference frame, interpolation, or metadata semantics.

## 2026-08-24 - GPU Pipeline and Strategy Selection

- Optimization stage: O1-O7 validation profiling.
- Code change: CUDA RGB residual, vectorized 4x4 partition, batched bilinear
  resize, candidate-only TinyRiskCNN, and pre-residual AFVT fast path.
- Correctness result: all stages preserve selected blocks on the validation
  smoke sequence; 26 unit/regression tests pass.
- Mean latency: Strategy A all-block partition is 0.795 ms at 37.5%;
  Strategy B direct gather is 0.847 ms.
- P95 latency: final validation path is 1.205 ms at 37.5%.
- Speedup: about 48.3x versus the frozen 38.404 ms selector baseline.
- Next action: freeze Strategy A and run cache-state equivalence.

## 2026-08-24 - Equivalence Smoke

- Optimization stage: Phase 4.6 validation smoke.
- Code change: full SegFormer/AFVT equivalence runner with frozen cache state.
- Correctness result: 57/57 frame-budget rows exactly match Round 3 selected
  blocks; maximum risk-score difference is `1.2636e-5`; max stale runs remain
  `3/2/1`.
- Mean latency: not used for the scientific smoke decision.
- P95 latency: not used for the scientific smoke decision.
- Speedup: not applicable.
- Next action: run one formal equivalence pass on the three frozen test videos.
