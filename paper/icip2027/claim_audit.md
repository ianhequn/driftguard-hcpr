# Round 10 Manuscript Claim Audit

Audit target: `main.tex`, compiled blind and publish PDFs, and the frozen Round 10 artifacts. The manuscript is a controlled characterization, not a new reuse architecture.

## Permitted claims and evidence

| Claim | Evidence |
| --- | --- |
| Whole-stage S4, S3--S4, and S2--S4 reuse can physically skip the cached suffix in the tested SegFormer pipelines. | `reports/round10_call_table.csv`; Table 1 |
| The same five-point reuse frontier is recovered for B0 and B1. | `reports/round10_pareto_audit.md`; `reports/round10_pareto_frontier.csv`; Table 2 and Fig. 2 |
| S4/FI-2 is the lowest-damage tested point, while S2--S4/FI-8 is the highest-speed tested point. | `reports/round10_boundary_quality_latency_table.csv`; `reports/round10_reuse_boundary_state.json` |
| Earlier boundaries expose more profiled suffix runtime, but realized speed is not proportional to opportunity. | `reports/round10_profile_table.csv`; `results/R10_E06_analysis/R10_F03_runtime_opportunity_vs_speed.pdf`; Table 3 and Fig. 4 |
| Cache memory must distinguish logical boundary-specific cache from allocated F2--F4 persistent state. | `reports/round10_cache_memory_audit.md`; `reports/round10_cache_memory_table.csv` |
| 100% refresh equivalence validates cache plumbing, not equality of stale features to current features. | `reports/round10_reuse_boundary_state.json`; Round 10 equivalence tests |

## Numeric spot checks

| Model / setting | mIoU drop (pp) | Paired speedup | 95% CI |
| --- | ---: | ---: | --- |
| B0 S4/FI-2 | 0.091 | 1.068 | [1.036, 1.100] |
| B0 S2--S4/FI-8 | 1.835 | 2.396 | [2.292, 2.492] |
| B1 S4/FI-2 | 0.118 | 1.111 | [1.061, 1.200] |
| B1 S2--S4/FI-8 | 2.715 | 2.282 | [2.086, 2.479] |

## Prohibited claims checked

- No first/novel/universal-optimality claim.
- No SOTA, edge-device, real-time guarantee, or cross-dataset generalization claim.
- No claim that S3--S4 reproduces TFC; it is explicitly labelled a vanilla baseline.
- No historical Round 5 RTX 4090 microbenchmark number is mixed with Round 10 RTX 4090 D results.
- No component-only speedup is reported as system speedup; the paper uses paired end-to-end latency.

## Result

`CLAIM AUDIT = PASS` for the final source and compiled PDFs. The word
“real-time” appears only in the description of cited RTFormer prior work, and
the manuscript explicitly limits its own evidence to the tested VSPW/B0/B1 and
RTX 4090 D protocol.
