# Round 11B Supported Claims

- Supported: whole-system HCPR transfer can produce measured CUDA-resident speedup on the evaluated RTX 3080 Ti when calibrated with second-GPU stage costs.
- Supported: B0 and B1 each have transferred HCPR-NewCost points that are non-dominated against the second-GPU fixed frontier.
- Supported: the physical stage-call and route counters show that stale suffix stages are skipped.
- Supported with scope: NewCost has strict measured Pareto improvements at B0/q65 versus OldCost and B1/q95 versus ProgressiveDriftOnly.
- Not supported: general hardware invariance, arbitrary GPUs, other model sizes, custom-kernel acceleration, or a claim that profile-only k transfer is deployment calibration.
- Recalibration: required when the target GPU changes; Round 11A 4090D costs are retained only as the OldCost ablation.
