from pathlib import Path

import cv2
import numpy as np

from driftguard.utils.video import get_video_info, iter_video_frames


def test_video_reader_round_trip(tmp_path: Path):
    video_path = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (32, 24),
    )
    assert writer.isOpened()
    for index in range(3):
        frame = np.full((24, 32, 3), index * 60, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    info = get_video_info(video_path)
    assert info.width == 32
    assert info.height == 24
    frames = list(iter_video_frames(video_path))
    assert len(frames) == 3
    assert frames[0][1].shape == (24, 32, 3)
