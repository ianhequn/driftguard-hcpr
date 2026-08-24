# DriftGuard Research Progress

## 2026-08-23

### Completed before long-cycle run

- Phase 0: environment and project baseline
- Phase 1: SegFormer-B0 image/video baseline
- Phase 2: synthetic temporal redundancy
- Phase 2.5: cloud RTX 4090 validation
- Phase 2.5: stage profiling
- Phase 2.5: decoder stale-feature sensitivity

### Phase 2.6

- Status: COMPLETED
- Objective: verify temporal redundancy on a small real-video collection.
- Configuration: SegFormer-B0, offsets `1,2,4,8,16`, global and block metrics, block grids `4x4` and `8x8`.
- Smoke test: completed.
- Result: 5 real videos, 32 frames/video, 160 frames total. All videos decoded successfully.
- Global Stage 4 cosine: `0.9949` at offset 1, `0.9528` at offset 16.
- Random Stage 4 cosine baseline: `0.9456`.
- Go/No-Go: GO. Adjacent-frame similarity exceeded random remote-pair similarity and decreased with offset.
- Next: sequential stale propagation, completed in Phase 2.7.

### Phase 2.7

- Result: 2,580 sequential stale records across 5 videos, 4 cuts, and 5 offsets.
- Corrected cache semantics: only the selected cached stage is stale; earlier and later stages remain fresh in this offline sequential model.
- Go/No-Go: GO.

### Phase 2.8

- Measured full latency: `20.084 ms`.
- Selected cache cut: `Cut 4`.
- Cut 4 latency: `1.937 ms`, approximate cache-path speedup `10.37x`.
- Cut 4 sequential logit relative L2: `0.0151` at offset 1 and `0.0199` at offset 2.
- Go/No-Go: GO. Cut 4 has the strongest measured latency utility with low short-term damage.

### Phase 3

- Selected cut: `Cut 4`.
- Result: mean feature drift rose from `0.0831` at age 1 to `0.2710` at age 16.
- Feature drift vs logit relative L2: Pearson `0.9512`, Spearman `0.9623`.
- Feature drift vs motion residual: Pearson `0.6870`, Spearman `0.7427`.
- Go/No-Go: GO. Cache age and cheap motion observables are informative enough to justify dataset generation.
- Next: generate video-level-split drift dataset.

### Phase 4

- Result: `10,320` block samples.
- Video-level split: train 3 videos / validation 1 video / test 1 video.
- Inputs: local RGB residual, cache age, previous prediction entropy, global residual, local residual variance.
- Go/No-Go: GO for estimator evaluation. Dataset generation completed without temporal leakage.

### Phase 5-6

- Baselines evaluated: Age Only, Residual Only, Linear Regression, Random Forest, Tiny MLP.
- Tiny MLP parameters: `737` baseline, `833` with interaction revision.
- Baseline Tiny MLP test metrics: MAE `0.0633`, RMSE `0.0836`, Pearson `0.5580`, Spearman `0.5561`.
- Best simple baseline: Residual Only, MAE `0.0391`, RMSE `0.0533`, Pearson `0.6299`, Spearman `0.7620`.
- Revision 1: `log1p` target, Tiny MLP Pearson `0.5162`.
- Revision 2: train-label P95 clipping, Tiny MLP Pearson `0.5345`.
- Revision 3: residual-age/global-age/log-age interactions, Tiny MLP Pearson `0.5444`.
- Tiny MLP overhead: mean `0.206 ms` per batch of 16, `0.0129 ms` per sample on RTX 4090.
- Go/No-Go: NO-GO for the learned DriftGuard estimator. Three controlled revisions did not beat the cheap RGB residual baseline.
- Scientific stop: do not run Phase 7 refresh simulation because the proposed learned predictor has no demonstrated incremental value.

### Round 1 Metadata

- Selected cache stage: `Stage 4`.
- Selected cache cut: `Cut 4`.
- Round 1 status: scientifically stopped after Phase 6 NO-GO.
- Round 2 is maintained in independent `results/R2_*` directories and does not overwrite Round 1 artifacts.
