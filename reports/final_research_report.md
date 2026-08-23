# DriftGuard First Research Loop

## Scientific Conclusion

**NO-GO for the current learned DriftGuard estimator.**

The real-video temporal redundancy hypothesis is supported, and feature drift is strongly related to task damage. However, the proposed tiny learned estimator did not improve over the cheap RGB residual baseline after three controlled revisions. Continuing to refresh-policy simulation would not establish incremental value for the proposed method.

## Evidence

| Phase | Result |
|---|---|
| Real redundancy | 5 videos, 32 frames/video; adjacent Stage 4 cosine `0.9949`, offset 16 `0.9528`, random baseline `0.9456` |
| Cache cut | Cut 4 selected; measured cache path `1.937 ms` vs full `20.084 ms` |
| Drift vs age | Stage 4 drift `0.0831` at age 1, `0.2710` at age 16 |
| Drift/task correlation | Pearson `0.9512`, Spearman `0.9623` for feature drift vs logit relative L2 |
| Dataset | 10,320 samples, video-level split 3/1/1 |
| Residual Only | Pearson `0.6299`, Spearman `0.7620`, MAE `0.0391` |
| Tiny MLP | Pearson `0.5580`, Spearman `0.5561`, MAE `0.0633` |
| Estimator overhead | `0.206 ms` per batch of 16 on RTX 4090 |

## Failed Revisions

1. `log1p` target transform: Pearson `0.5162`.
2. Train-label P95 clipping: Pearson `0.5345`.
3. Residual/age interaction features: Pearson `0.5444`.

All test metrics were computed on a held-out video. The invalid first sequential run caused by a cache-prefix implementation bug was preserved separately and excluded from conclusions.

## Next Decision

Do not enter Real Partial Computation or deployment work under the current estimator definition. A future research round would need to redesign the predictor or label/task coupling first, rather than adding CUDA or TensorRT engineering.
