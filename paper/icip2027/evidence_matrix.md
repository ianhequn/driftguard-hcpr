# DriftGuard ICIP 2027 Evidence Matrix

This matrix is the source-of-truth gate for every quantitative claim in the manuscript. Values are copied from committed artifacts; no new experiment is introduced by the paper draft.

| ID | Claim used in paper | Value | Primary source | Confidence | Planned use |
| --- | --- | --- | --- | --- | --- |
| E01 | VSPW evaluation scope | 343 validation videos, 24,392 RGB-GT pairs, 124 classes | `reports/round7_restart_final_report.md`; `reports/round8_resume_gt_benchmark.md` | High | Setup and limitation |
| E02 | B0 Full official-style mIoU | 0.30129196806407454 | `results/R7R_E02_full_gt/metrics.json`; `reports/round7_restart_final_report.md` | High | Main table |
| E03 | B0 FI-2 mIoU and drop | 0.3003732475378488; 0.091872052622574 pp | `results/R7R_E03_fi2_gate/metrics.json`; `reports/round7_restart_final_report.md` | High | Main table |
| E04 | B0 FI-2 repeated speed | 1.092377 +/- 0.075728x; 95% CI [1.033718, 1.151035]; 5 repetitions | `reports/round71_final_report.md`; `reports/round71_latency_robustness.md` | High | Main table and robustness text |
| E05 | B0 Stage 4 share | 0.23975421800348004 of full latency | `results/R71_E02_profile/profile_repetitions.json` | High | Profile table |
| E06 | B0 F4 cache footprint | 0.125 MB, BF16 F4 cache | `results/R8R_E10_cross_scale/cross_scale_table.csv`; `reports/round8_resume_cross_scale.md` | High | Profile table |
| E07 | B1 Full official-style mIoU | 0.37159793127621504 | `results/R8R_E06_full_validation/metrics.json`; `reports/round8_resume_gt_benchmark.md` | High | Main table |
| E08 | B1 FI-2 mIoU and drop | 0.3704133728540334; 0.11845584221816252 pp | `results/R8R_E10_cross_scale/cross_scale_table.csv`; `reports/round8_resume_gt_benchmark.md` | High | Main table |
| E09 | B1 FI-2 repeated speed | 1.020278 +/- 0.067943x; 95% CI [0.851484, 1.189071]; 3 repetitions | `reports/round8_resume_latency.md`; `results/R8R_E10_cross_scale/cross_scale_table.csv` | High | Cross-scale discussion |
| E10 | B1 FI-8 speed and quality cost | 1.224525 +/- 0.159141x; 1.4352972785053764 pp drop | `results/R8R_E10_cross_scale/cross_scale_table.csv`; `reports/round8_resume_cross_scale.md` | High | Trade-off discussion |
| E11 | B1 Stage 4 share | 0.18990660459834086 of full latency | `results/R8R_E10_cross_scale/stage_profile.json` | High | Profile table |
| E12 | B1 F4 cache footprint | 0.25 MB, BF16 F4 cache | `results/R8R_E10_cross_scale/stage_profile.json` | High | Profile table |
| E13 | Whole-stage refresh equivalence, B0 | 20 validation frames; max abs, mean abs and relative L2 all zero | `reports/round7_restart_equivalence.md`; `results/R7R_E00_preflight/equivalence.json` | High | Correctness |
| E14 | Whole-stage refresh equivalence, B1 | 20 validation frames; max abs, mean abs and relative L2 all zero; allclose true | `results/R8R_E07_reuse_equivalence/equivalence.json`; `reports/round8_resume_final_report.md` | High | Correctness |
| E15 | B1 fresh reload consistency | mIoU difference 3.51e-7 | `reports/round8_resume_final_report.md`; `reports/round8_resume_frozen_model.md` | High | Reproducibility |
| E16 | Training protocol | 40,000 steps; BF16; batch 8; seed 42; AdamW; lr 6e-5; weight decay 0.01; warmup 800; RTX 4090 | `results/R76R_E02_training/run_config.json`; `reports/round8_resume_training.md` | High | Setup |
| E17 | Latency protocol | 200 warmup and 1,000 CUDA-event measurements; CUDA-resident input; sequential cache state | `results/R71_E02_profile/profile_repetitions.json`; `reports/round71_latency_robustness.md`; `reports/round8_resume_latency.md` | High | Setup |
| E18 | B0 frozen checkpoint identity | SHA256 `8b673cb6bc173f3e986da8446c2c3debacc77779109b4916b230af36f197f17b` | `results/R76R_E04_freeze/freeze_summary.json`; `reports/round7_restart_final_report.md` | High | Reproducibility |
| E19 | Round 5 fine-grained partial result | Exact partial computationally pointless; approximate PyTorch route did not provide a usable system result | `reports/round5_final_report.md` | High | Motivation only |
| E20 | Round 6 structured reuse result | Whole Stage 4 dense refresh/reuse was a strong engineering GO; dynamic stage policy was NO-GO | `reports/round6_final_report.md` | High | Motivation and method |
| E21 | B0 formal refresh execution | Full: 24,392 Stage 4 calls; FI-2: 12,289; decoder remained per-frame | `reports/round7_restart_final_report.md` | High | Actual compute avoidance |
| E22 | Quality metric semantics | VSPW validation labels use the frozen official normalization; no pseudo-ground-truth | `reports/round7_restart_equivalence.md`; `reports/round7_restart_final_report.md` | High | Setup and limitation |
| E23 | Actual refresh ratios | B0/B1 FI-2: 12,289 / 24,392 = 50.381%; B1 FI-8: 3,181 / 24,392 = 13.041% | `reports/round8_resume_gt_benchmark.md`; `reports/round7_restart_final_report.md` | High | Main table and method |
| E24 | Fine-grained negative microbenchmark | Dense Stage4 4.366 ms; 37.5% cached-context route 6.662 ms | `reports/round5_final_report.md`; `reports/round5_progress.md` | High | Motivation only; separately scoped |

## Claim exclusions

- The old single-run B0 1.232x value is not used as the headline result.
- Round 9 Cityscapes recovery was blocked by storage/sequence availability and is not an experiment or a cross-dataset result.
- No claim is made for all SegFormer variants, all datasets, real-time operation, edge hardware, or a public VSPW test-set score.
