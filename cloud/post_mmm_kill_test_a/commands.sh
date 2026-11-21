#!/usr/bin/env bash
set -euo pipefail

# Upload this directory together with the exact POSTMMM-R1 source commit.
# These commands are plans only until checkpoint and dataset paths are filled.
python preflight.py --dry-run --model-key B0 --model-dir /PATH/TO/B0 --dataset-root /PATH/TO/VSPW --repo-root /PATH/TO/DriftGuard --output /tmp/postmmm-r1/preflight_b0.json
python run_fixed_quality.py --dry-run --model-key B0 --model-dir /PATH/TO/B0 --dataset-root /PATH/TO/VSPW --repo-root /PATH/TO/DriftGuard --output-root /tmp/postmmm-r1
python run_fixed_latency.py --dry-run --model-key B0 --model-dir /PATH/TO/B0 --dataset-root /PATH/TO/VSPW --repo-root /PATH/TO/DriftGuard --output-root /tmp/postmmm-r1
python run_hcpr_quality.py --dry-run --model-key B0 --model-dir /PATH/TO/B0 --dataset-root /PATH/TO/VSPW --repo-root /PATH/TO/DriftGuard --output-root /tmp/postmmm-r1
python run_hcpr_latency.py --dry-run --model-key B0 --model-dir /PATH/TO/B0 --dataset-root /PATH/TO/VSPW --repo-root /PATH/TO/DriftGuard --output-root /tmp/postmmm-r1
