# DriftGuard Round 7.6 Restart Final Report

## Final decision

```text
ROUND 7.6 RESTART = MODEL TRAINING SUCCESS
BENCHMARK READINESS = READY FOR ROUND 7 RESTART
```

The historical result remains unchanged:

```text
ROUND 7.6 = MODEL TRAINING NO-GO
```

This report records a new clean restart under the resolved Round 7.65
contract. No temporal reuse, FI-2/FI-3/FI-4/FI-6/FI-8, latency benchmark, or
Round 7 benchmark was run.

## Required answers

| Question | Result |
|---|---|
| Q1 Canonical contract used? | Yes: official normalization, then outside 0..123 -> 255 |
| Q2 Full-domain preflight? | PASS: train 197253, val 24392, remaining invalid 0 |
| Q3 Initialization SHA256? | `6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9` |
| Q4 Initialization audit? | 206 tensors loaded; classifier weight/bias only reinitialized |
| Q5 100-step smoke/reload? | PASS |
| Q6 Throughput / peak VRAM? | 21.667658 images/s; 4.534920 GB |
| Q7 Training steps? | 40000 |
| Q8 Epoch equivalent? | 1.62228204387259 |
| Q9 Best checkpoint? | Step 40000 |
| Q10 Best validation mIoU? | 0.3012895793848641 |
| Q11 Frozen checkpoint? | `/root/autodl-tmp/models/segformer-b0-vspw124` |
| Q12 Frozen model SHA256? | `8b673cb6bc173f3e986da8446c2c3debacc77779109b4916b230af36f197f17b` |
| Q13 Fresh reload mIoU? | 0.30129196806407454 |
| Q14 Reload vs best? | PASS; absolute delta 2.38867921042996e-06 |
| Q15 Output classes? | Strictly 124; forward shape `[1,124,128,128]` |
| Q16 Class alignment? | 124 / 124 VERIFIED |
| Q17 Raw253 handling? | PASS: `253 -> 252 -> 255`, excluded |
| Q18 New un-audited raw label? | No; full canonical scan passed |
| Q19 NaN/Inf/collapse? | No NaN/Inf; four never-predicted classes are recorded as long-tail risk, not global collapse |
| Q20 Structured reuse compatibility? | DIRECT |
| Q21 Round 7 restart ready? | Yes, authorized but not executed |

## Training cost and convergence

The clean run started from step 0 and completed 40000 optimizer steps, seeing
320000 images. Mean step time was 0.3692138758 s, and the training-only wall
time was approximately 29181.731 s (`8.1060 h`). The eight in-run full
validation gates consumed 4.6372 h; the complete training process including
those gates was 8.7449 h. The independent frozen reload validation consumed
2250.533 s (`0.6251 h`). The historical failed-run cost of 0.5183 active GPU
hours is not included in these values.

Validation mIoU increased as follows:

| Step | mIoU | Pixel accuracy |
|---:|---:|---:|
| 5000 | 0.1401837287 | 0.6428920673 |
| 10000 | 0.2272186686 | 0.6810247584 |
| 15000 | 0.2585681025 | 0.6910281717 |
| 20000 | 0.2860315142 | 0.6985434779 |
| 25000 | 0.2860492710 | 0.6978795120 |
| 30000 | 0.2902821459 | 0.6977386746 |
| 35000 | 0.2979340000 | 0.7023752585 |
| 40000 | 0.3012895794 | 0.7030939285 |

## Label contract and class coverage

The official VSPW normalization was applied before canonical range handling.
Raw label 253 was not assigned a semantic class: it maps to normalized 252
and then to ignore 255, excluded from loss and evaluation. No PNG dataset
files were changed. The fresh frozen validation reproduced the selected metric
and verified the class mapping.

The final prediction histogram has four never-predicted IDs (43, 47, 95,
113) and three rare IDs (4, 72, 73). This is an explicit long-tail coverage
risk for later benchmark interpretation. It does not invalidate the restart
gate because training remained numerically stable, all target classes were
present, the output head is 124-way, and the reload/full-validation contract
passed.

## Static architecture audit

The frozen model measured the expected SegFormer-B0 shapes:

| Stage | Shape |
|---:|---|
| F1 | `[1,32,128,128]` |
| F2 | `[1,64,64,64]` |
| F3 | `[1,160,32,32]` |
| F4 | `[1,256,16,16]` |

The decoder consumes F1-F4 and its classifier emits 124 channels. Therefore
the static structured-reuse compatibility result is `DIRECT`. This is only an
architecture compatibility result; temporal reuse was not executed.

## Evidence

- `results/R76R_E00_preflight/`
- `results/R76R_E01_smoke/`
- `results/R76R_E02_training/`
- `results/R76R_E03_validation/`
- `results/R76R_E04_freeze/`
- `results/R76R_E05_frozen_validation/`
- `results/R76R_E06_architecture_audit/`

The 20-frame QA manifest is `results/R76R_E06_architecture_audit/qa_manifest.json`;
the RGB/GT/prediction image triplets remain on the cloud experiment host.

## Stop boundary

The only authorized next activity is the separately controlled Round 7
ground-truth VSPW temporal-reuse benchmark. It is not started automatically.
Do not interpret this result as evidence that FI-2 maintains mIoU or that
temporal reuse works on VSPW.
