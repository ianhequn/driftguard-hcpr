"""Global and streaming temporal feature similarity measurements."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median, quantiles, stdev
from typing import Iterable

import torch

from .feature_blocks import RollingFeatureBuffer, block_metrics


def _validate_pair(current: torch.Tensor, reference: torch.Tensor) -> None:
    if not isinstance(current, torch.Tensor) or not isinstance(reference, torch.Tensor):
        raise TypeError("Feature pairs must be torch.Tensor objects.")
    if current.shape != reference.shape:
        raise ValueError(
            "Feature shape mismatch: "
            f"current={tuple(current.shape)}, reference={tuple(reference.shape)}"
        )
    if current.numel() == 0:
        raise ValueError("Feature tensors must not be empty.")
    if not torch.is_floating_point(current) or not torch.is_floating_point(reference):
        raise TypeError("Feature tensors must use a floating-point dtype.")


def cosine_similarity(
    current: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Return global cosine similarity after flattening the feature pair."""

    _validate_pair(current, reference)
    current_flat = current.detach().reshape(-1).float()
    reference_flat = reference.detach().reshape(-1).float()
    denominator = current_flat.norm(p=2) * reference_flat.norm(p=2)
    value = torch.dot(current_flat, reference_flat) / (denominator + eps)
    return float(value.clamp(-1.0, 1.0).item())


def relative_l2_distance(
    current: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Return ||current-reference||_2 / (||current||_2 + eps)."""

    _validate_pair(current, reference)
    current_float = current.detach().float()
    reference_float = reference.detach().float()
    numerator = (current_float - reference_float).norm(p=2)
    denominator = current_float.norm(p=2)
    return float((numerator / (denominator + eps)).item())


def _percentiles(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    quartiles = quantiles(values, n=4, method="inclusive")
    return quartiles[0], quartiles[2]


def summarize_pair_rows(rows: list[dict]) -> list[dict]:
    """Aggregate global pair rows by stage and temporal offset."""

    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["stage"]), int(row["offset"]))].append(row)

    summary: list[dict] = []
    for (stage, offset), group in sorted(grouped.items()):
        cosine_values = [float(item["global_cosine"]) for item in group]
        l2_values = [float(item["relative_l2"]) for item in group]
        cosine_p25, cosine_p75 = _percentiles(cosine_values)
        summary.append(
            {
                "stage": stage,
                "offset": offset,
                "num_pairs": len(group),
                "cosine_mean": mean(cosine_values),
                "cosine_std": stdev(cosine_values) if len(group) > 1 else 0.0,
                "cosine_median": median(cosine_values),
                "cosine_p25": cosine_p25,
                "cosine_p75": cosine_p75,
                "l2_mean": mean(l2_values),
                "l2_std": stdev(l2_values) if len(group) > 1 else 0.0,
                "l2_median": median(l2_values),
            }
        )
    return summary


def analyze_feature_stream(
    feature_frames: Iterable[tuple[int, list[torch.Tensor]]],
    offsets: Iterable[int],
    block_grid: int = 4,
    video_id: str = "stream",
) -> dict[str, object]:
    """Compare a stream of stage features with a bounded rolling buffer.

    `feature_frames` yields `(frame_id, [stage1, stage2, stage3, stage4])`.
    Feature tensors may be on CPU or CUDA; callers can move them to CPU before
    yielding when GPU memory is more important than comparison throughput.
    """

    normalized_offsets = sorted({int(offset) for offset in offsets})
    if not normalized_offsets or any(offset <= 0 for offset in normalized_offsets):
        raise ValueError("offsets must contain positive integers.")
    if block_grid <= 0:
        raise ValueError("block_grid must be a positive integer.")

    max_offset = max(normalized_offsets)
    buffer = RollingFeatureBuffer(max_size=max_offset + 1)
    pair_rows: list[dict] = []
    block_rows: list[dict] = []
    heatmap: dict[str, object] | None = None
    processed_frames = 0

    for frame_id, current_features in feature_frames:
        if len(current_features) != 4:
            raise ValueError(
                f"Expected four stage features, received {len(current_features)}."
            )
        for stage_index, feature in enumerate(current_features, start=1):
            if feature.ndim not in (3, 4):
                raise ValueError(
                    f"Stage {stage_index} must be CHW or BCHW, got rank {feature.ndim}."
                )

        for offset in normalized_offsets:
            reference_frame = int(frame_id) - offset
            reference_features = buffer.get(reference_frame)
            if reference_features is None:
                continue
            for stage_index, (current, reference) in enumerate(
                zip(current_features, reference_features),
                start=1,
            ):
                pair_rows.append(
                    {
                        "video_id": video_id,
                        "current_frame": int(frame_id),
                        "reference_frame": reference_frame,
                        "offset": offset,
                        "stage": stage_index,
                        "global_cosine": cosine_similarity(current, reference),
                        "relative_l2": relative_l2_distance(current, reference),
                        "block_grid": block_grid,
                    }
                )
                stage_blocks = block_metrics(
                    current,
                    reference,
                    grid_size=block_grid,
                )
                for item in stage_blocks:
                    block_rows.append(
                        {
                            "video_id": video_id,
                            "current_frame": int(frame_id),
                            "reference_frame": reference_frame,
                            "offset": offset,
                            "stage": stage_index,
                            "block_row": item["block_row"],
                            "block_col": item["block_col"],
                            "cosine": item["cosine"],
                            "relative_l2": item["relative_l2"],
                        }
                    )
                if (
                    heatmap is None
                    and offset == 1
                    and stage_index == 1
                ):
                    heatmap = {
                        "current_frame": int(frame_id),
                        "reference_frame": reference_frame,
                        "stage": stage_index,
                        "grid_size": block_grid,
                        "values": [
                            item["cosine"] for item in stage_blocks
                        ],
                    }

        buffer.add(int(frame_id), current_features)
        processed_frames += 1

    return {
        "pair_rows": pair_rows,
        "block_rows": block_rows,
        "summary_rows": summarize_pair_rows(pair_rows),
        "heatmap": heatmap,
        "processed_frames": processed_frames,
        "buffer_max_size": buffer.max_size,
        "buffer_peak_size": buffer.peak_size,
    }
