# DriftGuard Round 2 Final Report

## Decision

**STRONG GO for the learned task-aware refresh selector.**

This is a project-internal result for the tested dataset, model, cache semantics, and offline refresh simulation. It is not a claim of real partial-forward speedup.

## Dataset

| Item | Result |
|---|---:|
| Complete real videos | 21 |
| Train / validation / test | 15 / 3 / 3 videos |
| Inventory window | 48 frames/video |
| Motion groups | 7 low / 7 medium / 7 high |
| Test motion coverage | low, medium, high |
| Split rule | video-level |
| Cache cut | Stage 4 / Cut 4 |
| Spatial blocks | 4x4 = 16 |
| Cache ages | 1, 2, 4, 8, 16 |

Synthetic video was excluded from Round 2 train, validation, and test.

## Task Damage Labels

The main continuous label was `logit_relative_l2` from single-block Stage 4 stale-feature interventions. Thresholds were computed from training videos only:

| Threshold | Value |
|---|---:|
| P70 | 0.0084835 |
| P80 | 0.0114374 |
| P90 | 0.0177729 |

## Predictor

| Model | AUROC | AUPRC | Spearman | Parameters |
|---|---:|---:|---:|---:|
| Residual Mean | 0.9231 | 0.7672 | 0.6956 | 0 |
| Residual P90 | 0.9086 | 0.7451 | 0.6551 | 0 |
| Logistic Regression | 0.9443 | 0.7918 | 0.7494 | N/A |
| Random Forest | 0.9367 | 0.7649 | 0.7049 | N/A |
| TinyRiskCNN | 0.9504 | 0.8003 | 0.7768 | 5,865 |
| Old Tiny MLP, historical | 0.9580 | 0.8252 | 0.8442 | 737 |

The old MLP is reported unchanged as a historical classification baseline. It was not retuned; its stronger classification result does not undo Round 1's regression NO-GO.

TinyRiskCNN latency for all 16 blocks was `0.585 ms` mean, `0.628 ms` P95 on the RTX 4090.

## Ablation

| Input | AUPRC |
|---|---:|
| Patch only | 0.7370 |
| Patch + age | 0.8159 |
| Patch + age + entropy | 0.7671 |
| Patch + all metadata | 0.8003 |

## Fixed-Budget Refresh

At refresh ratios 25%, 37.5%, and 50%, mean test-video logit relative L2 was:

| Refresh ratio | Random | Residual Mean | Residual P90 | TinyRiskCNN | Age | Oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 25% | 0.02301 | 0.02999 | 0.03156 | **0.02017** | 0.01809 | 0.01654 |
| 37.5% | 0.01767 | 0.02320 | 0.02405 | **0.01581** | 0.01378 | 0.01220 |
| 50% | 0.01406 | 0.01722 | 0.01839 | **0.01285** | 0.01068 | 0.00925 |

TinyRiskCNN beat Residual Mean, Residual P90, and Random on all 3 held-out videos at all three key ratios. Age was the strongest simple policy in this simulation.

## Final Answer

The spatial residual patch contains task-relevant information that a scalar residual baseline does not fully capture. With cache age, TinyRiskCNN produced consistent fixed-budget gains and stayed below the 1 ms/frame predictor overhead target.

Next phase recommendation only: **Real Partial Computation + Accuracy/Latency Pareto**. Do not execute that phase automatically in this round.
