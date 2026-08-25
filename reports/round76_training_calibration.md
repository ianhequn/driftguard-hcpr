# DriftGuard Round 7.6 Training Calibration

## Selected Configuration

```text
GPU: NVIDIA GeForce RTX 4090 24 GB
precision: BF16
optimizer: AdamW
learning rate: 6e-5
weight decay: 0.01
schedule: polynomial decay, power 0.9
warmup: 800 / 40,000 steps (2%)
gradient clipping: 1.0
batch size: 8
gradient accumulation: 1
effective batch: 8
workers: 8
pin memory: true
persistent workers: true
prefetch factor: 2
```

## Batch Calibration

| Batch | Mean step time | Images/s | Peak allocated VRAM |
|---:|---:|---:|---:|
| 2 | 0.1221 s | 16.38 | 1.19 GB |
| 4 | 0.1987 s | 20.13 | 2.30 GB |
| 6 | 0.2889 s | 20.77 | 3.42 GB |
| 8 | 0.3482 s | 22.98 | 4.53 GB |

The 500-step batch-8 run remained finite and reduced interval loss from about
`4.82` to `3.34`. Observed GPU utilization during calibration was about 89%,
and device memory use reported by `nvidia-smi` was about 6.5 GB.

## Time Budget

```text
estimated 10k training time: 0.97 hours, excluding validation
estimated 40k training time: 3.87 hours, excluding validation
estimated hours per 10k steps: 0.97
attempted validation before semantic hard stop: approximately 0.113 hours
```

The measured training cost supports the 10k pilot and 40k main target. The
Round 7.5 provisional 12-48 GPU-hour range is superseded by this calibration.
No complete validation duration was obtained because the first full pass hit
the frozen dataset-integrity stop condition after approximately 6.8 minutes.
