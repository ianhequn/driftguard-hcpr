# DriftGuard Round 8 Resume: Cross-Scale Generalization

| Model | Policy | Full mIoU | Policy mIoU | Drop pp | Repeated Speedup | Stage4 Fraction | F4 Cache |
|---|---|---:|---:|---:|---:|---:|---:|
| B0 | FI-2 | 0.301292 | 0.300373 | 0.0919 | 1.0924 +/- 0.0757x | 23.975% | 0.125 MB |
| B1 | FI-2 | 0.371598 | 0.370413 | 0.1185 | 1.0203 +/- 0.0679x | 18.991% | 0.250 MB |
| B1 | FI-4 | 0.371598 | 0.366066 | 0.5532 | 1.0413 +/- 0.1571x | 18.991% | 0.250 MB |
| B1 | FI-8 | 0.371598 | 0.357245 | 1.4353 | 1.2245 +/- 0.1591x | 18.991% | 0.250 MB |

Whole-Stage4 reuse transfers from B0 to B1 as a measurable accuracy/latency trade-off, but the primary FI-2 speedup becomes weaker at B1 because Stage4 is a smaller fraction of the measured B1 full forward and GPU timing variability is material.

The claim is limited to SegFormer-B0/B1 under the frozen VSPW validation protocol. It is not generalized to all SegFormer models, other datasets, or edge hardware.
