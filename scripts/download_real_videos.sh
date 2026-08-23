#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/real_videos}"
mkdir -p "$ROOT"
cd "$ROOT"

download() {
  local name="$1"
  local url="$2"
  echo "[$(date -Is)] downloading $name"
  wget -c --tries=5 --waitretry=5 -O "$name" "$url"
  echo "[$(date -Is)] completed $name"
}

download people-detection.mp4 \
  https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4
download car-detection.mp4 \
  https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4
download bottle-detection.mp4 \
  https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/bottle-detection.mp4
download one-by-one-person-detection.mp4 \
  https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/one-by-one-person-detection.mp4
download person-bicycle-car-detection.mp4 \
  https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4

ls -lh
