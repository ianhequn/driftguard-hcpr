# DriftGuard Round 8 Resume: B1 Training

## Protocol

- Source: `nvidia/segformer-b1-finetuned-ade-512-512` recovered at the recorded mirror revision.
- Dataset: frozen official VSPW train/validation split, VSPW-124 label contract.
- Optimizer: AdamW, learning rate `6e-5`, weight decay `0.01`.
- Schedule: poly power `0.9`, warmup ratio `0.02`, gradient clipping `1.0`.
- Precision: BF16; physical batch `8`; seed `42`; clean optimizer step range `0 -> 40,000`.
- Checkpoints: 5k cadence; full validation at 10k, 20k, 30k, 40k.

## Calibration

The 500-step batch-8 calibration measured `0.3733934290 s/step`, `21.4251 images/s`, and `5.4146 GB` peak VRAM. No OOM or non-finite loss occurred.

## Validation Curve

| Step | mIoU | Pixel Accuracy | Validation Seconds |
|---:|---:|---:|---:|
| 10,000 | 0.2988457653 | 0.7213631786 | 2241.9567 |
| 20,000 | 0.3549961238 | 0.7407368149 | 1958.1028 |
| 30,000 | 0.3715116431 | 0.7485014599 | 2113.3245 |
| 40,000 | 0.3715982824 | 0.7490073203 | 2057.2351 |

Step 40,000 was selected by mIoU. Total training wall time was `23091.05 s`, including validation and checkpoint operations. The model did not require an automatic extension beyond 40k.
