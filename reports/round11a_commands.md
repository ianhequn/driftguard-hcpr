# Round 11A Commands

All formal commands were executed in `/root/autodl-tmp/DriftGuard` with `/root/miniconda3/bin/python` on the single RTX 4090 D instance. Exact command lines and parameters are preserved in the cloud screen logs and the JSON artifacts.

1. `python scripts/run_round11a_preflight.py --dataset-root data/raw/VSPW --b0-dir /root/autodl-tmp/models/segformer-b0-vspw124 --b1-dir /root/autodl-tmp/models/segformer-b1-vspw124 --output results/R11A_E00_preflight/preflight.json`
2. `python scripts/run_round11a_calibration.py ... --output-dir results/R11A_E02_calibration/b0`
3. `python scripts/run_round11a_quality.py ... --output-dir results/R11A_E03_b0_quality`
4. `python scripts/run_round11a_latency.py ... --output results/R11A_E04_b0_latency/latency.json`
5. `python scripts/run_round11a_calibration.py ... --output-dir results/R11A_E05_b1/calibration`
6. `python scripts/run_round11a_quality.py ... --output-dir results/R11A_E05_b1/quality`
7. `python scripts/run_round11a_latency.py ... --output results/R11A_E05_b1/latency/latency.json`
8. `python scripts/measure_round11a_controller.py ... --output results/R11A_E06_analysis/controller_{b0,b1}.json`
9. `python scripts/finalize_round11a.py --root .`

Each calibration used seed 42, 32 training videos, labels unused, 200 warmup iterations, 1000 CUDA-event measurements, and 3 repetitions. Formal latency used 200 warmup iterations, 1000 measurements, 5 repetitions, sequential cache state, and an immediate paired Full baseline.
