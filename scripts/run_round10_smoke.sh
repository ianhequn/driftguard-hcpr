#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/DriftGuard
for model in b0 b1; do
  model_dir="/root/autodl-tmp/models/segformer-${model}-vspw124"
  output_dir="results/R10_E02_smoke/${model}"
  /root/miniconda3/bin/python scripts/run_round10_quality.py \
    --model-dir "$model_dir" \
    --dataset-root /root/autodl-tmp/DriftGuard/data/raw/VSPW \
    --output "$output_dir/full.json" --boundary full \
    --limit-videos 2 --limit-frames 10 --workers 2
  for boundary in s4 s3_s4 s2_s4; do
    for interval in 2 4 8; do
      /root/miniconda3/bin/python scripts/run_round10_quality.py \
        --model-dir "$model_dir" \
        --dataset-root /root/autodl-tmp/DriftGuard/data/raw/VSPW \
        --output "$output_dir/${boundary}_fi${interval}.json" \
        --boundary "$boundary" --interval "$interval" \
        --limit-videos 2 --limit-frames 10 --workers 2
    done
  done
done
echo SMOKE_PASS
