# ICASSP 2027 HCPR Number Audit

This audit covers the final source `paper/icip2027/Hequn_Lan_ICASSP2027_HCPR.tex`.
No new experiment was run for the manuscript rewrite.

| Manuscript claim | Paper value | Frozen source | Check |
| --- | ---: | --- | --- |
| VSPW validation videos | 343 | `reports/round11b_state.json`, `dataset_contract.videos` | PASS |
| VSPW RGB/GT pairs | 24,392 | `reports/round11b_state.json`, `dataset_contract.frames` | PASS |
| VSPW classes | 124 | `reports/round11b_state.json`, `dataset_contract.classes` | PASS |
| Ignore label | 255 | `reports/round11b_state.json`, `dataset_contract.ignore` | PASS |
| B0 / RTX 4090 D / q20 speedup | 1.229x | `reports/round11a_hcpr_quality_latency_table.csv` | PASS |
| B0 / RTX 4090 D / q20 damage | 0.073 pp | same, rounded from 0.072958 | PASS |
| B1 / RTX 4090 D / q20 speedup | 1.323x | same, rounded from 1.322921 | PASS |
| B1 / RTX 4090 D / q20 damage | 0.115 pp | same, rounded from 0.114877 | PASS |
| B0 / RTX 3080 Ti / q20 speedup | 1.354x | `reports/round11b_cross_hardware_table.csv` | PASS |
| B0 / RTX 3080 Ti / q20 damage | 0.0726 pp | same, exact 0.0725866 | PASS |
| B0 / RTX 3080 Ti / q65 speedup | 1.838x | same, exact 1.8382115 | PASS |
| B0 / RTX 3080 Ti / q65 damage | 0.428 pp | same, exact 0.4279365 | PASS |
| B0 / RTX 3080 Ti / q95 speedup | 2.010x | same, exact 2.0100914 | PASS |
| B0 / RTX 3080 Ti / q95 damage | 1.394 pp | same, exact 1.3939727 | PASS |
| B1 / RTX 3080 Ti / q20 speedup | 1.344x | same, exact 1.3436238 | PASS |
| B1 / RTX 3080 Ti / q20 damage | 0.1193 pp | same, exact 0.1192988 | PASS |
| B1 / RTX 3080 Ti / q65 speedup | 1.849x | same, exact 1.8493508 | PASS |
| B1 / RTX 3080 Ti / q65 damage | 0.875 pp | same, exact 0.8746196 | PASS |
| B1 / RTX 3080 Ti / q95 speedup | 2.093x | same, exact 2.0929475 | PASS |
| B1 / RTX 3080 Ti / q95 damage | 2.179 pp | same, exact 2.1789208 | PASS |
| B0 q65 NewCost vs OldCost | 1.838x / 0.428 pp vs 1.801x / 0.441 pp | `results/R11B_E06_analysis/hardware_ablation_table.csv` | PASS |
| B1 q95 NewCost vs DriftOnly | 2.093x / 2.179 pp vs 2.042x / 2.681 pp | same | PASS |
| B0 controller overhead on 3080 Ti | 0.6431 ms, 5.819% | `reports/round11b_final_report.md` | PASS |
| B1 controller overhead on 3080 Ti | 0.6890 ms, 6.248% | same | PASS |
| B0 q20 physical calls | 24392 / 14094 / 13792 / 13781 / 24392 | `reports/round11b_call_table.csv` | PASS |
| B1 q20 physical calls | 24392 / 13700 / 13669 / 13659 / 24392 | same | PASS |
| Maximum stale run | 7 | `reports/round11b_final_report.md`, age guard audit | PASS |

The paper uses rounded values only where the table is explicitly a compact
summary. The headline abstract values retain four decimal places for damage.

