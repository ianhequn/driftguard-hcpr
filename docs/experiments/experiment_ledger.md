# Experiment ledger

The ledger separates algorithm evidence from historical implementation details.

| Phase | Main question | Primary implementation | Current interpretation |
|---|---|---|---|
| Round 5 | Can partial Stage 4 computation pay off? | `driftguard/partial/` | Negative boundary; not part of the final claim |
| Round 6 | Can visual-risk / age policies select refreshes cheaply? | `driftguard/structured_reuse/policies.py` | Selector overhead and quality trade-offs require caution |
| Round 7/7.1 | Does fixed whole-Stage4 reuse physically save work? | fixed boundary pipeline | Strongest historical B0 baseline evidence |
| Round 7.6/8 | Can B0/B1 VSPW-124 models be trained and frozen? | `scripts/train_round76_segformer.py` | Historical cloud checkpoints; local weight bodies missing |
| Round 10 | What is the same-protocol fixed boundary matrix? | `scripts/run_round10_*.py` | Complete historical RTX 4090 D fixed set |
| Round 11A | Can drift and marginal hardware cost calibrate progressive routing? | `scripts/run_round11a_*.py` | Same-protocol HCPR source-device diagnostic |
| Round 11B | Does the controller transfer across hardware and cost variants? | `scripts/run_round11b_*.py` | Target HCPR set exists; fixed quality is not target-matched |
| POSTMMM-R1 | Does a clean device-matched fixed/HCPR comparison survive? | `cloud/post_mmm_kill_test_a/` | Prepared, not executed |

The authoritative caveat is
[`reports/post_mmm/pareto_consistency_audit.md`](../../reports/post_mmm/pareto_consistency_audit.md).
