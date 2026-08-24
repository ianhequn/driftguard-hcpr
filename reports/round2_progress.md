# DriftGuard Round 2 Progress

## 2026-08-24

### Research Pivot

- Status: RUNNING.
- Objective: replace feature-drift regression with task-damage risk prediction for Stage 4 / Cut 4 cache reuse.
- Historical constraint: the Round 1 scalar Tiny MLP remains frozen after its scientific NO-GO result.
- Primary evidence standard: at equal refresh budgets, TinyRiskCNN must reduce segmentation task damage relative to residual-only policies on video-level held-out test videos.

### Phase A: Dataset Expansion

- Status: COMPLETED.
- Result: 21 complete real videos, 15 train / 3 validation / 3 test, 48-frame inventory window.
- Motion distribution: 7 low / 7 medium / 7 high; test contains low, medium, and high motion.
- Data policy: synthetic video is excluded from all Round 2 train, validation, and test data.

### Phase B: Task-Damage Label Generation

- Status: COMPLETED.
- Result: 21 resumable per-video shards, with 3,344 samples per normal 48-frame video and 3,264 for the 47-frame clip.
- Intervention: Stage 4 / Cut 4, 4x4 blocks, cache ages `1,2,4,8,16`.
- Main continuous label: `logit_relative_l2`.
- Training-only thresholds: P70 `0.0084835`, P80 `0.0114374`, P90 `0.0177729`.
- Smoke test: passed on 2 videos, 16 frames, ages `1,4`.

### Phase C-E: Predictor

- Status: COMPLETED.
- Test threshold: training-video P80, unsafe positive ratio `0.1307`.
- TinyRiskCNN: 5,865 parameters; AUROC `0.9504`; AUPRC `0.8003`; Spearman `0.7768`.
- Residual Mean: AUROC `0.9231`; AUPRC `0.7672`; Spearman `0.6956`.
- Logistic Regression: AUROC `0.9443`; AUPRC `0.7918`; Spearman `0.7494`.
- Historical Tiny MLP is retained as a classification baseline; it was stronger than TinyRiskCNN on this label, but is not retuned.
- Predictor latency: all 16 blocks `0.585 ms` mean; P95 `0.628 ms`.

### Phase E Ablation

- Status: COMPLETED because TinyRiskCNN beat the primary residual baseline.
- Patch only: AUPRC `0.7370`.
- Patch + age: AUPRC `0.8159`.
- Patch + age + entropy: AUPRC `0.7671`.
- Patch + all metadata: AUPRC `0.8003`.
- Interpretation: spatial residual structure contributes, while cache age is the most useful metadata addition in this experiment.

### Phase F-G: Refresh Simulation

- Status: COMPLETED.
- Test videos: `asl_brother`, `asl_milk`, `asl_school`.
- Refresh ratios: `0.125, 0.25, 0.375, 0.5, 0.75, 1.0`.
- At 25% / 37.5% / 50%, TinyRiskCNN mean logit damage was `0.02017 / 0.01581 / 0.01285`.
- At the same ratios, Residual Mean was `0.02999 / 0.02320 / 0.01722`.
- TinyRiskCNN beat Residual Mean, Residual P90, and 20-seed Random on all 3 test videos at all three key ratios.
- Age policy was stronger than TinyRiskCNN; this is recorded as the strongest simple refresh baseline.
- Bootstrap: 1,000 video-level resamples; CNN-minus-baseline 95% CIs were below zero for all key comparisons.
- Final internal decision: `STRONG GO`.

### Planned Artifacts

- `results/R2_E00_video_dataset`: real-video inventory and video-level split.
- `results/R2_E01_task_damage_dataset`: Stage 4 single-block intervention labels.
- `results/R2_E02_predictor`: scalar baselines and TinyRiskCNN evaluation.
- `results/R2_E03_ablation`: conditional predictor ablations.
- `results/R2_E04_refresh_simulation`: fixed-budget policy comparison.
