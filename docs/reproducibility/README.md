# Reproducibility guide

## Artifact contract

Every formal result must record:

- model family and exact checkpoint SHA256;
- dataset split, video order, label mapping, and dataset manifest hash;
- git commit and Python/PyTorch/Transformers/CUDA versions;
- device identity and precision;
- protocol ID, timing warmup/measurement/repetition counts;
- absolute latency, speedup confidence interval, mIoU damage, route counts,
  and stage calls per frame.

## Historical checkpoints

The expected frozen model hashes are recorded, but the B0/B1 weight files were
not recovered locally. Do not substitute the ADE base checkpoint. See
[`docs/reviews/checkpoint_recovery_decision.md`](../reviews/checkpoint_recovery_decision.md).

## Clean next run

The next run is deliberately a new lineage:

```text
POSTMMM-R1
```

Use the two model configs and `configs/post_mmm/protocol.yaml`. Unknown fields
are marked `TODO_EVIDENCE_REQUIRED`; they must be filled from the actual cloud
run, not guessed from Round 10/11 metadata.

## What is not reproducible on CPU

CPU tests can validate state transitions and numerical semantics. They cannot
validate CUDA-resident latency, BF16 timing, device-specific stage costs, or a
GPU Pareto frontier.
