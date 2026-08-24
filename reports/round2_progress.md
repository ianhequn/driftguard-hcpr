# DriftGuard Round 2 Progress

## 2026-08-24

### Research Pivot

- Status: RUNNING.
- Objective: replace feature-drift regression with task-damage risk prediction for Stage 4 / Cut 4 cache reuse.
- Historical constraint: the Round 1 scalar Tiny MLP remains frozen after its scientific NO-GO result.
- Primary evidence standard: at equal refresh budgets, TinyRiskCNN must reduce segmentation task damage relative to residual-only policies on video-level held-out test videos.

### Phase A: Dataset Expansion

- Status: PENDING cloud connectivity.
- Target: at least 20 real consecutive-video clips, with low, medium, and high motion represented in each evaluation split.
- Data policy: synthetic video is excluded from all Round 2 train, validation, and test data.

### Planned Artifacts

- `results/R2_E00_video_dataset`: real-video inventory and video-level split.
- `results/R2_E01_task_damage_dataset`: Stage 4 single-block intervention labels.
- `results/R2_E02_predictor`: scalar baselines and TinyRiskCNN evaluation.
- `results/R2_E03_ablation`: conditional predictor ablations.
- `results/R2_E04_refresh_simulation`: fixed-budget policy comparison.
