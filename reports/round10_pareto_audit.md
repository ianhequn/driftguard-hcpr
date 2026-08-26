# Round 10 Pareto Audit

## Definition

Point A dominates point B when mean paired speedup is no smaller, mIoU
drop is no larger, and at least one inequality is strict. Full is retained
as the reference row but the reuse frontier is computed over the nine
non-Full boundary/interval settings for each model.

## B0

| Boundary | Interval | mIoU drop (pp) | Mean speedup |
|---|---:|---:|---:|
| S4 | FI-2 | 0.091024 | 1.067699 |
| S3-S4 | FI-2 | 0.183880 | 1.253641 |
| S2-S4 | FI-2 | 0.231304 | 1.505304 |
| S2-S4 | FI-4 | 0.825391 | 2.057414 |
| S2-S4 | FI-8 | 1.835033 | 2.395734 |

Dominated settings: S3-S4/FI-4, S3-S4/FI-8, S4/FI-4, S4/FI-8.
## B1

| Boundary | Interval | mIoU drop (pp) | Mean speedup |
|---|---:|---:|---:|
| S4 | FI-2 | 0.118205 | 1.111170 |
| S3-S4 | FI-2 | 0.282884 | 1.331454 |
| S2-S4 | FI-2 | 0.396708 | 1.553821 |
| S2-S4 | FI-4 | 1.280235 | 1.951985 |
| S2-S4 | FI-8 | 2.715460 | 2.282435 |

Dominated settings: S3-S4/FI-4, S3-S4/FI-8, S4/FI-4, S4/FI-8.

