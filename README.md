# DriftGuard

## Hardware-Calibrated Progressive Temporal Reuse for Video Segmentation

DriftGuard is a research codebase for reducing redundant computation in
frame-by-frame semantic segmentation. The central idea is not to mask the
output of a dense model after inference, but to make a physical routing
decision before expensive SegFormer stages are executed:

```text
current frame -> fresh prefix -> drift / hardware cost score
              -> {S2-S4, S3-S4, S4, Full}
              -> cached features + decoder -> logits
```

The repository contains the implementation, controlled experiments, audit
scripts, tests, and reproducibility metadata behind the DriftGuard study.
Large datasets and model weights are intentionally excluded.

## Algorithm contribution

The project is organized around four research mechanisms:

1. **Physical whole-stage reuse.** A skipped suffix is not computed and then
   discarded. Cached intermediate features are passed directly to the decoder.
2. **Hardware-calibrated routing.** Boundary drift is normalized by unlabeled
   calibration statistics and measured marginal stage cost:

   ```text
   q_j = drift_j / (scale_j * k_j)
   ```

3. **Progressive multi-boundary actions.** The controller can select `S2-S4`,
   `S3-S4`, `S4`, or `Full`, instead of using only one fixed refresh boundary.
4. **Temporal safety state.** Per-video cache reset, first-frame Full execution,
   reference refresh, and a maximum stale-run guard are explicit in the state
   machine and tested independently.

The research claim is intentionally scoped: this code demonstrates an
auditable whole-stage temporal-reuse controller. It does not currently claim
universal hardware superiority, temporal consistency improvement, or a valid
fixed-vs-HCPR Pareto frontier across mixed historical protocols.

## Repository map

```text
driftguard/
  structured_reuse/       fixed boundaries, HCPR controller, R11B variants
  models/                 SegFormer loading and stage feature access
  datasets/               VSPW dataset and label contract
  profiling/              CUDA/CPU profiling utilities
  metrics/                evaluation helpers
  analysis/               temporal drift and scheduler analyses
scripts/
  run_round10_*.py        controlled fixed-boundary study
  run_round11a_*.py       unlabeled calibration and HCPR source protocol
  run_round11b_*.py       cross-hardware transfer diagnostics
  audit_*.py              contract, provenance, and Pareto audits
tests/                    CPU unit tests for execution semantics
configs/post_mmm/         new POSTMMM-R1 reproduction contract
cloud/post_mmm_kill_test_a/
                          guarded cloud bundle for device-matched Kill Test A
docs/                     algorithm, experiment, and reproducibility notes
reports/                  lightweight historical evidence and audit outputs
results/                  selected experiment artifacts; no model weights
paper/                    manuscript sources and generated paper artifacts
```

The detailed implementation map is in
[`docs/algorithm/code_map.md`](docs/algorithm/code_map.md).

## Quickstart: CPU development

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
.\\.venv\\Scripts\\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\\.venv\\Scripts\\python.exe -m pip install pytest transformers pillow numpy pyyaml
```

Run the algorithm-semantic test suite:

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q `
  tests/test_hcpr_pipeline.py `
  tests/test_round11b_variants.py `
  tests/test_audit_mmm_pareto_consistency.py `
  tests/test_audit_checkpoint_inventory.py
```

These tests verify forced-Full equivalence, physical stage omission, age-guard
propagation, deterministic routing, variant cost semantics, and audit logic.

## GPU experiment workflow

Formal latency experiments require CUDA, BF16 support, the frozen checkpoint,
and the exact protocol metadata. The intended order is:

1. run environment and checkpoint preflight;
2. run Full/fixed quality and latency under one device contract;
3. calibrate drift scales and stage costs without labels;
4. run HCPR quality and latency with the same model, video order, and cache
   reset rule;
5. finalize only rows with the same model, device, quality protocol, timing
   protocol, checkpoint SHA256, and dataset manifest hash.

The prepared `POSTMMM-R1` bundle is documented in
[`cloud/post_mmm_kill_test_a/README.md`](cloud/post_mmm_kill_test_a/README.md).
Its entrypoints are fail-closed and support dry-run planning; they do not
silently reuse Round 10/11 latency numbers.

## Evidence and limitations

Historical Round 10/11 results are retained for audit, not presented as one
new clean Pareto set. The exact historical B0/B1 frozen weights were not found
in the local machine, archives, Git objects, or WSL files. The recovery
decision and candidate hashes are recorded in:

- [`docs/reviews/checkpoint_recovery_decision.md`](docs/reviews/checkpoint_recovery_decision.md)
- [`reports/post_mmm/checkpoint_inventory.md`](reports/post_mmm/checkpoint_inventory.md)
- [`reports/post_mmm/pareto_consistency_audit.md`](reports/post_mmm/pareto_consistency_audit.md)

No dataset or checkpoint is downloaded by the repository's default commands.

## Project status

`revision/mmm2027-post-review` is an evidence-first research branch. The next
scientifically valid step is a GPU `POSTMMM-R1` baseline, followed by the
device-matched Kill Test A. Paper claims should be updated only after those
artifacts pass the protocol checks.
