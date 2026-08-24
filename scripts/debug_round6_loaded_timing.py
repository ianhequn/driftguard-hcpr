"""Compare fixed and visual-style timing after loading the full test set."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_round6_formal_test import (  # noqa: E402
    block_scores,
    load_predictor,
    load_test_videos,
    make_specs,
    read_csv,
    sequence_samples,
)
from driftguard.models.segformer import load_segformer  # noqa: E402
from driftguard.structured_reuse.pipeline import StructuredReusePipeline  # noqa: E402


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


def timed_with_noop_events(fn):
    torch.cuda.synchronize()
    outer_start = torch.cuda.Event(enable_timing=True)
    outer_end = torch.cuda.Event(enable_timing=True)
    selector_start = torch.cuda.Event(enable_timing=True)
    selector_end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    outer_start.record()
    selector_start.record()
    selector_end.record()
    value = fn()
    outer_end.record()
    outer_end.synchronize()
    return (
        float(outer_start.elapsed_time(outer_end)),
        (time.perf_counter() - wall_start) * 1000.0,
        value,
    )


def main() -> None:
    device = torch.device("cuda")
    records = read_csv(
        PROJECT_ROOT / "results/R2_E00_video_dataset/video_inventory.csv"
    )
    processor, model = load_segformer(
        "/root/autodl-tmp/models/segformer-b0",
        device=device,
        local_files_only=True,
    )
    predictor, mean, std = load_predictor(
        PROJECT_ROOT / "results/R2_E03_ablation/patch_age.pt",
        device,
    )
    videos = load_test_videos(records, processor, device, 48)
    pipeline = StructuredReusePipeline(model)
    video = videos[0]
    pixel = video.pixel_values[1]
    current = video.rgb_gpu[1]
    previous = video.rgb_gpu[0]
    with torch.inference_mode():
        for _ in range(50):
            pipeline.forward(pixel, True, None)
    records_out = []
    _, reuse_state, _ = pipeline.forward(pixel, True, None)
    _, refresh_state, _ = pipeline.forward(pixel, True, None)
    with torch.inference_mode():
        for _ in range(50):
            records_out.append(
                timed_with_noop_events(
                    lambda: pipeline.forward(pixel, True, refresh_state)
                )
            )
        for _ in range(50):
            records_out.append(
                timed(lambda: pipeline.forward(pixel, False, reuse_state))
            )
        for _ in range(50):
            def visual_call():
                scores = block_scores(
                    predictor,
                    mean,
                    std,
                    current,
                    previous,
                    0,
                )
                del scores
                return pipeline.forward(pixel, True, None)
            records_out.append(timed(visual_call))
    for name, start, end in (
        ("fixed_refresh", 0, 50),
        ("fixed_reuse_uninitialized", 50, 100),
        ("visual_refresh", 100, 150),
    ):
        values = np.asarray(
            [[row[0], row[1]] for row in records_out[start:end]]
        )
        print(
            name,
            "event_mean",
            float(values[:, 0].mean()),
            "event_median",
            float(np.median(values[:, 0])),
            "wall_mean",
            float(values[:, 1].mean()),
            "wall_median",
            float(np.median(values[:, 1])),
        )

    state_by_video = {item.video_id: None for item in videos}
    sequence_values = []
    with torch.inference_mode():
        for item, frame_index in sequence_samples(videos, 100):
            state = state_by_video[item.video_id]
            if frame_index == 0:
                state = None
            event_ms, wall_ms, value = timed(
                lambda item=item, frame_index=frame_index, state=state:
                pipeline.forward(
                    item.pixel_values[frame_index],
                    True,
                    state,
                )
            )
            _, state, _ = value
            state_by_video[item.video_id] = state
            sequence_values.append((event_ms, wall_ms))
    values = np.asarray(sequence_values)
    print(
        "actual_sequence_full_refresh",
        "event_mean",
        float(values[:, 0].mean()),
        "event_median",
        float(np.median(values[:, 0])),
        "wall_mean",
        float(values[:, 1].mean()),
        "wall_median",
        float(np.median(values[:, 1])),
    )

    state_by_video = {item.video_id: None for item in videos}
    previous_by_video = {item.video_id: None for item in videos}
    dynamic_values = []
    with torch.inference_mode():
        for item, frame_index in sequence_samples(videos, 100):
            state = state_by_video[item.video_id]
            previous = previous_by_video[item.video_id]
            current = item.rgb_gpu[frame_index]
            if frame_index == 0:
                state = None
                previous = None

            def dynamic_call(
                item=item,
                frame_index=frame_index,
                state=state,
                previous=previous,
                current=current,
            ):
                if previous is not None:
                    scores = block_scores(
                        predictor,
                        mean,
                        std,
                        current,
                        previous,
                        0 if state is None else state.age,
                    )
                    del scores
                return pipeline.forward(
                    item.pixel_values[frame_index],
                    True,
                    state,
                )

            event_ms, wall_ms, value = timed(dynamic_call)
            _, state, _ = value
            state_by_video[item.video_id] = state
            previous_by_video[item.video_id] = current
            dynamic_values.append((event_ms, wall_ms))
    values = np.asarray(dynamic_values)
    print(
        "actual_sequence_visual_refresh",
        "event_mean",
        float(values[:, 0].mean()),
        "event_median",
        float(np.median(values[:, 0])),
        "wall_mean",
        float(values[:, 1].mean()),
        "wall_median",
        float(np.median(values[:, 1])),
    )


if __name__ == "__main__":
    main()
