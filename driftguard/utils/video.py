"""Small OpenCV helpers for streaming video experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int


def get_video_info(path: str | Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        return VideoInfo(
            path=str(path),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def iter_video_frames(
    path: str | Path,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield `(frame_id, RGB frame)` without loading the whole video."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_id = start_frame
    yielded = 0
    try:
        while max_frames is None or yielded < max_frames:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield frame_id, frame_rgb
            frame_id += 1
            yielded += 1
    finally:
        capture.release()
