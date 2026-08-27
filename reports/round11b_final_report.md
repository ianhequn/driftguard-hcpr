# DriftGuard Round 11B: Cross-Hardware Validation

## Decision

**ROUND 11B = CROSS-HARDWARE GO**

The primary result is a second-GPU validation of the frozen HCPR controller. No model, dataset split, precision, input resolution, decoder, or HCPR semantics were changed.

## Contract

- Second GPU: NVIDIA GeForce RTX 3080 Ti; CUDA 11.8; PyTorch 2.1.2+cu118; Transformers 4.44.2.
- Dataset: VSPW val, 343 videos, 24392 frames, 124 classes, ignore 255.
- Parent Round 11A commit: `4c6bc07b0f8fc78dd6c94d6a1d10f159c0f3221a`.

## Main Result

- B0 best low-damage point: HCPR-NewCost/q20, damage 0.072587 pp, paired speedup 1.354007x.
- B1 best low-damage point: HCPR-NewCost/q20, damage 0.119299 pp, paired speedup 1.343624x.
- Overall measured fastest primary point: B1 HCPR-NewCost/q95, 2.092948x.
- Controller overhead is measured in `controller_overhead_cross_device.csv`; it is not omitted from end-to-end timing.

## Controller Overhead

- B0: controller 0.643146 ms / profile Full 11.052737 ms = 5.819%.
- B1: controller 0.689013 ms / profile Full 11.028257 ms = 6.248%.

## Hardware Ablation

NewCost, OldCost, and ProgressiveDriftOnly are kept as separate variants. NewCost gives a strict improvement over OldCost at B0/q65, but the comparison is mixed across B1 and against DriftOnly. Therefore Claim A is supported, while the stronger universal hardware-calibration benefit claim is not.

## Required Answers

1. **Second GPU:** NVIDIA GeForce RTX 3080 Ti; compute capability [8, 6]; 11910 MB reported VRAM.
2. **Software match:** CUDA 11.8, PyTorch 2.1.2+cu118, and Transformers 4.44.2 match Round 11A; the driver differs (580.76.05 versus Round 11A 595.71.05), so this is not described as a pure driver-free hardware isolation.
3. **Profile differences:** B0 c2/c3/c4 = 2.603709/2.728337/2.288658 ms and k = 0.235571403/0.246847160/0.207067100; B1 c2/c3/c4 = 2.745045/2.733099/2.246174 ms and k = 0.248910164/0.247826878/0.203674400.
4. **B0 validity:** yes, 7 HCPR-NewCost points are non-dominated on the second-GPU Gate-A plot.
5. **B1 validity:** yes, 6 HCPR-NewCost points are non-dominated on the second-GPU Gate-A plot.
6. **Non-dominated counts:** B0 7; B1 6.
7. **Route shift:** replacing only k changes allocation modestly on B0 (maximum absolute shift 0.176 pp) and more visibly on B1 (1.337 pp); no shift was required for GO.
8. **NewCost versus OldCost:** yes locally at B0/q65, where NewCost improves both measured damage and speed; B1 and other anchors are mixed.
9. **NewCost versus DriftOnly:** NewCost generally gives lower damage at matched anchors, while DriftOnly can be faster at some points; there is no universal strict joint improvement.
10. **Quality budgets:** the non-interpolated fastest measured points are recorded in `quality_budget_table.csv`; OldCost is selected at several tight budgets, while B0/NewCost q65 wins the 0.5/1.0 pp rows.
11. **Controller overhead:** B0 0.643146 ms; B1 0.689013 ms.
12. **Overhead fraction:** 5.819%. for B0 and 6.248%. for B1; this is smaller than the corresponding Round 11A ratios.
13. **Physical savings:** yes; Stage 1 and decoder calls remain 24,392 while Stage 2/3/4 calls fall on reuse routes.
14. **Age guard:** maximum stale run remains exactly 7 for all tested variants and stages.
15. **Profile-only transfer:** supported for the evaluated RTX 3080 Ti, B0, and B1 with frozen Round 11A lambdas.
16. **Lambda recalibration:** not required; only target-device stage-cost profiling was added.
17. **Manuscript Claim A:** cross-device robustness is supported for the two tested SegFormer scales and the two tested GPUs.
18. **Manuscript Claim B:** only cautious lightweight hardware profiling language is supported; strong universal calibration benefit is not.
19. **Unsupported claim:** universal GPU portability or a universally causal hardware-normalization advantage.
20. **Manuscript timing:** rewriting may begin only after this Round 11B freeze, using the cautious Claim A/Claim B boundary.

## Limitations

This is a two-device validation on the frozen VSPW split. It supports transfer to the evaluated RTX 3080 Ti, not universal hardware invariance. A new target GPU requires measuring its stage-cost vector and rerunning the prescribed validation.

## Artifacts

See `results/R11B_E06_analysis/`, `reports/round11b_cross_hardware_table.csv`, and `reports/round11b_hardware_ablation_table.csv`.
