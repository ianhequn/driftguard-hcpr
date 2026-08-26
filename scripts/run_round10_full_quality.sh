#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/DriftGuard
DATA=/root/autodl-tmp/DriftGuard/data/raw/VSPW
for model in b0 b1; do
  MODEL=/root/autodl-tmp/models/segformer-${model}-vspw124
  OUT=results/R10_E03_quality/${model}
  /root/miniconda3/bin/python scripts/run_round10_quality.py --model-dir "$MODEL" --dataset-root "$DATA" --output "$OUT/full.json" --boundary full --workers 16
  for boundary in s4 s3_s4 s2_s4; do
    for interval in 2 4 8; do
      /root/miniconda3/bin/python scripts/run_round10_quality.py --model-dir "$MODEL" --dataset-root "$DATA" --output "$OUT/${boundary}_fi${interval}.json" --boundary "$boundary" --interval "$interval" --workers 16
    done
  done
done
echo QUALITY_MATRIX_PASS
