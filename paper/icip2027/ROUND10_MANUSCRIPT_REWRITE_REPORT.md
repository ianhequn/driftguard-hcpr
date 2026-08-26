# Round 10 Manuscript Rewrite Report

## Scope

The final manuscript is a controlled characterization of whole-stage
temporal reuse. It compares S4, vanilla S3--S4, and S2--S4 boundaries across
FI-2, FI-4, and FI-8 on frozen SegFormer-B0/B1 VSPW validation runs and paired
RTX 4090 D latency.

## Cache discrepancy resolution

`BoundaryCacheState` physically retains F2, F3, and F4 after refresh for every
active boundary. The old export therefore reported the same allocated cache
within a model. The corrected audit separates the two meanings:

| Model | Logical S4 | Logical S3--S4 | Logical S2--S4 | Allocated persistent |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.250 MiB | 0.875 MiB | 1.875 MiB | 1.875 MiB |
| B1 | 0.500 MiB | 1.750 MiB | 3.750 MiB | 3.750 MiB |

Counts use `numel() * element_size()` from the profile tensor shapes. No
quality, latency, or dataset experiment was rerun. The corrected CSV, profile
table, and final report expose `logical_cache_*` and `allocated_cache_*`.

## Pareto resolution

The dominance audit uses mean paired speedup and mIoU drop. Both B0 and B1
recover the same five-point frontier:

```text
S4/FI-2 -> S3-S4/FI-2 -> S2-S4/FI-2 -> S2-S4/FI-4 -> S2-S4/FI-8
```

This is reported as a bounded empirical ordering under the tested models and
GPU, not as a universal scheduling rule.

## Manuscript changes

- Replaced the previous S4-centered draft with a boundary x interval x model-scale study.
- Added explicit execution equations, decoder semantics, stage-call evidence, cache accounting, paired timing protocol, and limitations.
- Added the complete 18-setting result table, corrected profile/cache table, boundary Pareto figure, quality-refresh figure, and runtime-opportunity figure.
- Added explicit TFC positioning as related high-level feature reuse, without reproducing TFC alignment, fusion, or correction.
- Removed historical Round 5 numerical values from the quantitative evidence.
- Added claim and citation audits tied to repository artifacts.

## Final paper inventory

```text
Title: WHEN DOES TEMPORAL FEATURE REUSE ACCELERATE VIDEO SEMANTIC SEGMENTATION? A HARDWARE-AWARE STUDY OF REUSE BOUNDARIES
Abstract: 128 words
Technical pages: 4
Reference pages: 1
Total pages: 5
Figures: 4
Manuscript tables: 3
Verified references: 11 / 11
```

## Remaining human actions

Author identity, final ICIP 2027 template recheck, and human language/advisor
review remain TODO. No new experiment is required for this paper finalization.
