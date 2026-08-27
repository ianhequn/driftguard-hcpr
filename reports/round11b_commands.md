# Round 11B Reproduction Commands

```text
python scripts/run_round11b_preflight.py ...
python -m pytest -q
python scripts/run_round11b_profile.py ...
python scripts/run_round11b_gate.py ...
python scripts/run_round11b_quality.py ... --variant HCPR-NewCost
python scripts/run_round11b_latency.py ... --quantiles 20,65,95
python scripts/finalize_round11b.py --root .
```

All formal quality and latency runs use the frozen VSPW val split and the stated CUDA protocol.
