# Round 7 Restart: Ground-Truth Benchmark

## Scope

The benchmark uses the official VSPW validation split: 343 videos and 24,392 RGB-GT frame pairs. This is the only split used here because the public VSPW test split has no public ground truth. The same validation split was also used during frozen checkpoint selection in Round 7.6; therefore these numbers are a validation benchmark, not an independent test-set generalization claim.

## Full baseline gate

The measured Full mIoU is `0.30129196806407454`, exactly matching the frozen reference `0.30129196806407454`.

| Policy | mIoU | Drop (pp) | Pixel accuracy | Mean class accuracy | Actual refresh | Stage4 calls |
|---|---:|---:|---:|---:|---:|---:|
| Full | 0.301292 | 0.0000 | 0.703089 | 0.398897 | 100.000% | 24,392 |
| FI-2 | 0.300373 | 0.0919 | 0.702262 | 0.397778 | 50.381% | 12,289 |
| FI-3 | 0.298768 | 0.2524 | 0.700828 | 0.396132 | 33.691% | 8,218 |
| FI-4 | 0.296928 | 0.4364 | 0.699142 | 0.393797 | 25.508% | 6,222 |
| FI-6 | 0.292336 | 0.8956 | 0.695309 | 0.389484 | 17.264% | 4,211 |
| FI-8 | 0.289973 | 1.1319 | 0.691605 | 0.386296 | 13.041% | 3,181 |

The primary FI-2 quality gate passes with a 0.0919 percentage-point mIoU drop. Because the drop is far below the 5 pp hard-stop threshold, the remaining fixed-interval curve was executed.

## Per-class and long-tail note

`results/R7R_E05_pareto/full_vs_fi2_per_class.csv` contains the Full/FI-2 per-class IoU comparison. The known VSPW-124 long-tail issue from Round 7.6 remains visible: classes 43, 47, 95 and 113 were never predicted by the frozen checkpoint, while classes 4, 72 and 73 have very sparse support. These are checkpoint/data limitations and are not relabeled as a temporal-reuse gain.

## Accuracy interpretation

The Full/FI-2 comparison measures task damage against ground truth. It does not replace the prior temporal-reuse comparisons. Round 5 sparse partial-compute NO-GO and Round 6 dynamic-selector NO-GO remain frozen historical conclusions.
