# DriftGuard ICIP 2027 Paper Draft Report

## Status

```text
MANUSCRIPT SOURCE = COMPLETE
EVIDENCE AUDIT = COMPLETE
FIGURE/TABLE SOURCES = COMPLETE
PDF COMPILATION = BLOCKED BY ENVIRONMENT
```

The manuscript is a source draft, not a submission-ready PDF. The requested
blind and publish output paths are reserved under `output/`, but no PDF is
claimed until a TeX toolchain compiles and audits both variants.

## Title

`HARDWARE-AWARE STRUCTURED TEMPORAL REUSE FOR EFFICIENT VIDEO SEMANTIC SEGMENTATION`

## Scientific result

The paper's primary result is whole-Stage4 temporal reuse on the frozen VSPW
validation protocol. SegFormer-B0 FI-2 reaches a repeated paired end-to-end
speedup of 1.0924 +/- 0.0757x over five repetitions, with a 95% interval of
[1.0337, 1.1510], while the global validation mIoU drop is 0.0919 percentage
points. The B1 transfer preserves a 0.1185-point FI-2 drop but has weak,
uncertain speed gain. B1 FI-8 reaches 1.2245x with a 1.4353-point drop.

Stage4 is 23.98% of measured B0 full latency and 18.99% of B1 full latency.
This supports the paper's hardware-aware conclusion: the system benefit is
bounded by the fraction of work that can actually be skipped.

## Evaluation scope

The ground-truth evaluation uses 343 VSPW validation videos, 24,392 RGB-GT
pairs, and 124 classes. The public VSPW test labels are unavailable, so the
paper makes no independent test-set claim. Round 9 Cityscapes recovery was
blocked by storage/sequence availability and is excluded from all results.

## Figures and tables

- `figures/method_overview.pdf`
- `figures/accuracy_latency_pareto.pdf`
- `figures/stage4_runtime_share.pdf`
- `tables/main_results.csv`
- `tables/stage4_profile.csv`

Each generated asset has a JSON sidecar naming its repository sources.

## References

Ten bibliography entries are included and listed in `reference_audit.csv`.
They were checked against official proceedings, CVF Open Access, or the
authors' arXiv record. No fabricated entries were added to meet a target count.

## Required remaining actions

1. Provide the final author list, affiliation, email, funding, and
   acknowledgments if applicable; see `AUTHORS_TODO.md`.
2. Compile both `blind.tex` and `publish.tex` with a TeX installation.
3. Run the ICIP PDF checker and inspect page count, embedded fonts, metadata,
   anonymous text, and reference-only page policy.
4. Recheck the final ICIP 2027 Author Kit before submission.

## Explicit non-claims

The draft does not claim state of the art, universal model/dataset
generalization, edge-device validation, real-time guarantees, Cityscapes
performance, or speedup from block-sparse/custom-kernel execution.
