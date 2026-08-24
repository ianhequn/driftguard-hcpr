#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/real_videos}"
mkdir -p "$ROOT"
cd "$ROOT"

BASE="https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master"

download() {
  local name="$1"
  local url="$2"
  if [[ -s "$name" ]]; then
    echo "[skip] $name"
    return
  fi
  echo "[download] $name"
  wget -c --tries=5 --waitretry=5 -O "$name.part" "$url"
  mv "$name.part" "$name"
}

# Five clips were used in Round 1 and are retained. These additional clips
# come from the same public CC-BY sample repository, but remain separate
# video files and are split at the video level.
for name in \
  bolt-detection.mp4 \
  bolt-multi-size-detection.mp4 \
  classroom.mp4 \
  driver-action-recognition.mp4 \
  face-demographics-walking-and-pause.mp4 \
  face-demographics-walking.mp4 \
  fruit-and-vegetable-detection.mp4 \
  head-pose-face-detection-female-and-male.mp4 \
  head-pose-face-detection-female.mp4 \
  head-pose-face-detection-male.mp4 \
  store-aisle-detection.mp4 \
  worker-zone-detection.mp4
do
  download "$name" "$BASE/$name"
done

# ASL clips add indoor human motion diversity without synthetic frames.
for name in again bird book brother eat help hungry learn milk night no please school sister sorry student thanks walk want yes; do
  download "asl_${name}.mkv" "$BASE/asl_gestures/${name}.mkv"
done

rm -f classroom.mp4.partial classroom.mp4.part
ls -lh
