# DriftGuard Round 8 Resume: B1 Ground-Truth Benchmark

All policies were evaluated on the complete frozen VSPW validation split: `343 videos / 24,392 frames`.

| Policy | Refresh Ratio | Stage4 Calls | mIoU | Pixel Accuracy | Mean Class Accuracy | Drop vs Full |
|---|---:|---:|---:|---:|---:|---:|
| Full | 100.000% | 24,392 | 0.3715979313 | 0.7490037485 | 0.4737976622 | 0 pp |
| FI-2 | 50.381% | 12,289 | 0.3704133729 | 0.7481159586 | 0.4724140273 | 0.1185 pp |
| FI-4 | 25.508% | 6,222 | 0.3660656587 | 0.7453135007 | 0.4677666287 | 0.5532 pp |
| FI-8 | 13.041% | 3,181 | 0.3572449585 | 0.7379001071 | 0.4591362248 | 1.4353 pp |

The exact per-class and per-video values are stored in `results/R8R_E08_gt_curve/`. FI-2 is the primary quality point: its global official-style mIoU loss is only `0.1185 percentage points`.

Round 3's historical AFVT-vs-Age warning remains preserved: the earlier 37.5% AFVT operating point had approximately 5.6% pixel disagreement relative to Age. This B1 whole-stage fixed-interval result is a separate mechanism and does not erase that warning.
