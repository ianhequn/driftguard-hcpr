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

## Limitations

This is a two-device validation on the frozen VSPW split. It supports transfer to the evaluated RTX 3080 Ti, not universal hardware invariance. A new target GPU requires measuring its stage-cost vector and rerunning the prescribed validation.

## Artifacts

See `results/R11B_E06_analysis/`, `reports/round11b_cross_hardware_table.csv`, and `reports/round11b_hardware_ablation_table.csv`.
