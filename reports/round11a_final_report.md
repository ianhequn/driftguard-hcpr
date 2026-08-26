# DriftGuard Round 11A: HCPR Feasibility

## Final decision

**ROUND 11A = STRONG GO**

## Scope

HCPR was evaluated as progressive whole-stage reuse on frozen VSPW validation videos. Stage 1 is always fresh, the decoder always runs, and only the Stage 2/3/4 continuation boundary is controlled. No model, data split, resolution, precision, decoder, or lower-level kernel was changed.

## Frozen contract

- GPU: NVIDIA GeForce RTX 4090 D; CUDA 11.8; PyTorch 2.1.2+cu118; Transformers 4.44.2; BF16; 512x512; batch 1.
- VSPW validation: 343 videos, 24,392 RGB/GT pairs, 124 classes, ignore 255.
- Cache resets at video boundaries; first frame is Full; formal timing uses 200 warmup, 1,000 CUDA-event measurements, five repetitions, and an immediate paired Full baseline.

## Method and calibration

HCPR maintains C2/C3/C4, pooled references R1_to_2/R2_to_3/R3_to_4, and age2/age3/age4. At each boundary it computes FP32 GAP/cosine drift, divides by an unlabeled 32-training-video p90 scale, normalizes by measured downstream stage cost k, and continues only when q > lambda or the age guard (7) forces refresh. Lambda values are the frozen 20/35/50/65/80/90/95% score quantiles; no validation mIoU was used for routing calibration.

## Correctness and physical proof

The cloud regression suite passed 109 tests with 2 expected skips. HCPR forced Full is exactly equivalent to native SegFormer on the tiny-model contract; forced S4/S3-S4/S2-S4 matches Round 10 semantics. Per-frame stage-call deltas prove that a stopped suffix is not executed, and decoder calls equal frame count. The full call table is `reports/round11a_hcpr_call_table.csv`.

## B0 and B1 results

B0 Gate A and B1 Gate B each contain seven HCPR candidates that are not dominated by the five remeasured Round-10 fixed frontier points under the full-precision (mIoU-drop, paired-speedup) audit. Exact points and values are in `reports/round11a_hcpr_quality_latency_table.csv`; audits are in `results/R11A_E06_analysis/b0/pareto_audit.json` and `results/R11A_E06_analysis/b1/pareto_audit.json`.

## Controller overhead

The isolated controller diagnostic measured 0.7089 ms/frame on B0 and 0.7548 ms/frame on B1 for the three boundary decisions. This cost includes GPU pooling, cosine, normalization, scalar synchronization, and branching; it is also included in formal end-to-end latency.

## Required questions

1. New B0 Pareto point: yes (7 points).
2. New B1 Pareto point: yes (7 points).
3. Route mix: recorded per lambda in the quality table and raw JSON.
4. Removed calls: recorded exactly per model and lambda in the call table.
5. Controller overhead: measured above.
6. End-to-end acceleration after synchronization: yes for the HCPR operating points with paired speedup > 1.
7. Hardware normalization versus DriftOnly: reported in the 20-video diagnostic ablation table; this one-GPU result is not a portability claim.
8. Age guard: maximum ages and stale runs are recorded; the guard is 7.
9. Lowest-damage points: listed in state JSON under `best_points`.
10. Highest-speed points: listed in state JSON under `best_points`.
11. HCPR dominates no fixed point under the strict audit; it instead adds non-dominated operating points between fixed regions.
12. Qualitative effect reproduces on B0 and B1: yes.
13. Round 11B evidence: sufficient to justify preparation, not automatic execution.

## Failure cases and limitations

The fixed frontier latency was remeasured on the same process; several fixed points vary materially across runs, which is why historical Round-10 speedups were not substituted. The controller adds measurable synchronization overhead. Only one GPU and two frozen model scales were tested. No claim of portability or universal optimality is made.

## Final next action

Freeze Round 11A and prepare Round 11B cross-hardware validation. Do not start it automatically.
