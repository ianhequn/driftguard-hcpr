# Reproducibility Contract

The paper reports the frozen DriftGuard artifacts already produced by the project. The benchmark boundary starts from a CUDA-resident model input tensor and ends when segmentation logits are ready. It includes fresh Stages 1-3, the fixed-interval refresh decision, dense Stage 4 refresh or final-F4 reuse, and the decoder. It excludes video decoding, disk I/O, pageable CPU H2D, visualization, and CSV writing.

The whole-stage cache stores only the final Stage 4 feature F4. A video starts with a compulsory Stage 4 refresh; a refresh replaces F4 and resets cache age; reuse returns the cached F4 and skips Stage 4; the decoder still runs for every frame. Stage 1-3 stay fresh on every frame.

The primary policy is FixedInterval-2 (FI-2). Its logical refresh ratio is 50%, while the measured B0 VSPW validation sequence executed 12,289 refreshes over 24,392 frames (50.381%). The reported speed is paired sequential CUDA timing, with 200 warmup iterations, 1,000 measurements, and repeated runs as specified by each source report.

The B0 checkpoint is frozen by SHA256 and the B1 checkpoint is frozen after a clean 40,000-step VSPW-compatible training run. The public VSPW test labels are unavailable; all ground-truth numbers in this paper are validation-split numbers and are labeled accordingly.
