# DriftGuard Round 7.6 Restart Progress

## Scope

This restart trains and freezes a VSPW-compatible SegFormer-B0 under the
Round 7.65 canonical label contract. The historical Round 7.6 result remains
`MODEL TRAINING NO-GO`; this document records a new clean run.

Temporal reuse, AFVT, FI-2, latency benchmarking, and Round 7 benchmarking
were not executed.

## Completed gates

- Source-of-truth reports and code read.
- Full-domain canonical preflight passed: train 197253 masks, validation
  24392 masks, remaining invalid labels 0.
- Initialization SHA256 matched the historical source.
- 18 regression tests passed and 2 were skipped as expected.
- 100-step BF16 smoke passed with checkpoint save/reload.
- Clean training ran from step 0 through step 40000.
- Best validation checkpoint selected at step 40000.
- Frozen model exported to the dedicated VSPW-124 directory.
- Stage 1-4 and decoder architecture audit passed with DIRECT structured-reuse
  compatibility.
- 20-frame RGB/GT/prediction QA export completed.
- Full cloud regression suite passed: 94 passed, 2 skipped.

## Final gate

Fresh-process full validation completed over 24392 validation frames. Reload
mIoU was `0.30129196806407454`, versus the selected training value
`0.3012895793848641`; the absolute difference was `2.38867921042996e-06`.
Class alignment was `124 / 124`.

The restart result is `MODEL TRAINING SUCCESS`. Benchmark readiness is
`READY FOR ROUND 7 RESTART`, but Round 7 was not started.

## Stop boundary

After the frozen reload gate and reports are complete, stop. Do not enter
Round 7 GT temporal-reuse benchmarking in this task.
