# DriftGuard Round 7.6 Restart Training Record

## Configuration

| Field | Value |
|---|---|
| Architecture | SegFormer-B0 |
| Classes | 124 |
| Image size | 512 x 512 |
| Dataset | VSPW official train/val split |
| Label contract | official normalization, then IDs outside 0..123 -> 255 |
| Ignore index | 255 |
| Batch / accumulation | 8 / 1 |
| Precision | BF16 |
| Optimizer | AdamW |
| Learning rate | 6e-5 |
| Weight decay | 0.01 |
| Warmup | 800 steps of 40000 |
| Schedule | polynomial power 0.9 |
| Seed | 42 |
| Workers | 8 |
| Optimizer steps | 40000 |
| Images seen | 320000 |
| Epoch equivalent | 1.62228204387259 |

## Initialization

The source was `/root/autodl-tmp/models/segformer-b0` with SHA256
`6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9`.
The restart loaded 206 compatible tensors, with only the classifier weight and
bias reinitialized. The encoder was trainable throughout training.

## Performance and validation curve

- Mean training step time: 0.3692138758 s.
- Training throughput: 21.667658 images/s.
- Peak allocated VRAM: 4.534920 GB.
- Total process elapsed time: 31481.779 s.
- Training-only wall time through step 40000: approximately 29181.731 s.
- Full validation elapsed time across eight gates: 4.637221 h.

Validation mIoU:

| Step | mIoU | Pixel accuracy |
|---:|---:|---:|
| 5000 | 0.1401837287 | 0.6428920673 |
| 10000 | 0.2272186686 | 0.6810247584 |
| 15000 | 0.2585681025 | 0.6910281717 |
| 20000 | 0.2860315142 | 0.6985434779 |
| 25000 | 0.2860492710 | 0.6978795120 |
| 30000 | 0.2902821459 | 0.6977386746 |
| 35000 | 0.2979340000 | 0.7023752585 |
| 40000 | 0.3012895794 | 0.7030939285 |

The selected checkpoint is the 40000-step checkpoint.
