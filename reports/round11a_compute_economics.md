# Round 11A Compute Economics

## Measured boundary costs

The hardware calibration measured the downstream continuation costs used by HCPR on the RTX 4090 D. B0 costs were approximately 1.826 ms (Stage 2), 1.896 ms (Stage 3), and 1.532 ms (Stage 4), with a 7.793 ms full-model reference. B1 costs were approximately 1.785 ms, 1.813 ms, and 1.474 ms, with a 7.563 ms full-model reference.

These values calibrate routing scores; they are not a claim that a stage cost is constant across all clocks, devices, or workloads.

## End-to-end result

The formal boundary is CUDA-resident input to segmentation logits, including Stage 1, the HCPR controller, the selected continuation, and the decoder. Paired Full baselines were measured immediately after each policy. The resulting full-precision HCPR speedups are recorded in `reports/round11a_hcpr_quality_latency_table.csv` and the raw five-repetition latency files.

The lowest-damage HCPR point reaches 1.229x on B0 with 0.073 mIoU percentage-point drop and 1.323x on B1 with 0.115 percentage-point drop. Higher-refresh suppression reaches 1.836x on B0 and 1.919x on B1, with larger measured quality damage.

## Amdahl boundary

Only whole-stage continuation reuse is evaluated. Stage 1 and the decoder remain on every frame, so system speedup is bounded by the portion of work removed from Stage 2/3/4 and by controller/cache overhead. The reported result is measured end-to-end speedup, not a stage-only or FLOPs-only estimate.
