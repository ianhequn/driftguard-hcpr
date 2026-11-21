# Post-MMM Pareto Consistency Audit

## 结论

当前论文的 fixed-vs-HCPR non-dominance 结论不能作为严格同 protocol 结论接受。Round 10 fixed policy、Round 11A HCPR、Round 11B target fixed/HCPR 的来源链被脚本显式拆开；跨 protocol 的数字只作为 diagnostic，不进入 strict Pareto set。

## 来源链

| 集合 | 设备 | quality source | latency source | strict 状态 |
|---|---|---|---|---|
| Round 10 fixed | RTX 4090 D | `results/R10_E03_quality` | `results/R10_E04_latency` | 可作为内部同 protocol 集合 |
| Round 11A HCPR | RTX 4090 D | `reports/round11a_hcpr_quality_latency_table.csv` | 同一 CSV 的 Round 11A 汇总 | 可作为内部同 protocol 集合；不能直接与 Round 10 fixed 合并 |
| Round 11B fixed | RTX 3080 Ti | Round 10 quality | `results/R11B_E04_latency` | INVALID/MIXED_PROTOCOL |
| Round 11B HCPR | RTX 3080 Ti | `results/R11B_E03_transfer` | `results/R11B_E04_latency` | 可作为 HCPR-only 集合 |

## 论文文字检查

- Table 2 source-device anchor 与 artifact 一致：`True`；target-device anchor 与 artifact 一致：`True`。
- Table 3 的代表性 fixed 数值与 Round 10 artifact 一致：`True`。
- Table 3 表格块显式标 GPU：`False`；当前为 `False`。
- Table 5 cost profile 数值与 Round 11A/R11B profile artifact 一致：`True`。
- Table 6 ablation 数值全部与 Round 11B 汇总一致：`False`；不一致项：`[{'model': 'B0', 'variant': 'ProgressiveDriftOnly', 'quantile': 95, 'artifact': '1.778/2.065'}]`。
- 论文写有 4090 D 全 7 点 non-dominated：`True`。
- 论文写有 target B0 7/7、B1 6/7：`True`。
- 在 diagnostic 数值重算中，source 4090 D B0 q90 的 dominator：`S2-S4/FI4, HCPR-NewCost/q80`。
- source 4090 D B0 q95 的 dominator：`S2-S4/FI4`。

## 解释

B0 source q90/q95 的 fixed dominator 已足以推翻“4090 D 全 7 点 non-dominated”。这不是重新测量得出的新结果，而是对仓库已有 artifact 的一致性重算。target 的 7/7、6/7 计数只能描述混合来源的 diagnostic comparison，不能替代同设备、同 quality protocol 的 fixed sweep。

## 机器可读明细

每个点都保留 model/device/method/boundary/interval/quantile、quality/latency source、两个 protocol ID、damage、speedup 与 speedup CI；见同目录 JSON。

### source_4090D

| model | HCPR | dominators |
|---|---|---|
| B0 | HCPR-NewCost/q20 | none |
| B0 | HCPR-NewCost/q35 | none |
| B0 | HCPR-NewCost/q50 | none |
| B0 | HCPR-NewCost/q65 | none |
| B0 | HCPR-NewCost/q80 | none |
| B0 | HCPR-NewCost/q90 | S2-S4/FI4, HCPR-NewCost/q80 |
| B0 | HCPR-NewCost/q95 | S2-S4/FI4 |
| B1 | HCPR-NewCost/q20 | none |
| B1 | HCPR-NewCost/q35 | none |
| B1 | HCPR-NewCost/q50 | none |
| B1 | HCPR-NewCost/q65 | none |
| B1 | HCPR-NewCost/q80 | none |
| B1 | HCPR-NewCost/q90 | S2-S4/FI4 |
| B1 | HCPR-NewCost/q95 | S2-S4/FI4 |

### target_3080Ti

| model | HCPR | dominators |
|---|---|---|
| B0 | HCPR-NewCost/q20 | none |
| B0 | HCPR-NewCost/q35 | none |
| B0 | HCPR-NewCost/q50 | none |
| B0 | HCPR-NewCost/q65 | none |
| B0 | HCPR-NewCost/q80 | none |
| B0 | HCPR-NewCost/q90 | none |
| B0 | HCPR-NewCost/q95 | none |
| B1 | HCPR-NewCost/q20 | none |
| B1 | HCPR-NewCost/q35 | none |
| B1 | HCPR-NewCost/q50 | none |
| B1 | HCPR-NewCost/q65 | none |
| B1 | HCPR-NewCost/q80 | none |
| B1 | HCPR-NewCost/q90 | none |
| B1 | HCPR-NewCost/q95 | none |
