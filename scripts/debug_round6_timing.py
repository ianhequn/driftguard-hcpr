"""Small same-process timing sanity for Round 6 refresh/reuse."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.models.segformer import load_segformer, prepare_inputs  # noqa: E402
from driftguard.structured_reuse.pipeline import StructuredReusePipeline  # noqa: E402
from driftguard.utils.video import iter_video_frames  # noqa: E402


def timed(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    value = fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), (time.perf_counter() - wall_start) * 1000.0, value


def main() -> None:
    device = torch.device("cuda")
    processor, model = load_segformer(
        "/root/autodl-tmp/models/segformer-b0",
        device=device,
        local_files_only=True,
    )
    frame = next(
        iter_video_frames(
            PROJECT_ROOT / "data/real_videos/asl_brother.mkv",
            max_frames=1,
        )
    )[1]
    pixel = prepare_inputs(
        processor,
        Image.fromarray(frame, mode="RGB"),
        device,
    )["pixel_values"]
    pipeline = StructuredReusePipeline(model)
    state = None
    for _ in range(20):
        with torch.inference_mode():
            _, state, _ = pipeline.forward(pixel, True, state)
    torch.cuda.synchronize()
    values = []
    with torch.inference_mode():
        for index in range(50):
            event_ms, wall_ms, _ = timed(
                lambda: pipeline.forward(
                    pixel,
                    True,
                    None,
                )
            )
            values.append(("refresh", event_ms, wall_ms))
        _, state, _ = pipeline.forward(pixel, True, None)
        for index in range(50):
            event_ms, wall_ms, _ = timed(
                lambda: pipeline.forward(
                    pixel,
                    False,
                    state,
                )
            )
            values.append(("reuse", event_ms, wall_ms))
    direct = []
    with torch.inference_mode():
        for _ in range(50):
            event_ms, wall_ms, _ = timed(
                lambda: model(pixel_values=pixel, return_dict=True)
            )
            direct.append(("direct", event_ms, wall_ms))
    for name, records in (
        ("refresh", [row for row in values if row[0] == "refresh"]),
        ("reuse", [row for row in values if row[0] == "reuse"]),
        ("direct", direct),
    ):
        array = np.asarray([[row[1], row[2]] for row in records])
        print(
            name,
            "event_mean",
            float(array[:, 0].mean()),
            "event_median",
            float(np.median(array[:, 0])),
            "wall_mean",
            float(array[:, 1].mean()),
            "wall_median",
            float(np.median(array[:, 1])),
        )


if __name__ == "__main__":
    main()
