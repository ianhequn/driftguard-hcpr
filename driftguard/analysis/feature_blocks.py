"""Spatial block operations and bounded feature buffering."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch


def _as_chw(feature: torch.Tensor) -> torch.Tensor:
    if not isinstance(feature, torch.Tensor):
        raise TypeError("Feature must be a torch.Tensor.")
    if feature.ndim == 4:
        if feature.shape[0] != 1:
            raise ValueError(
                "Block analysis supports BCHW tensors only with batch size 1."
            )
        return feature[0]
    if feature.ndim == 3:
        return feature
    raise ValueError(
        f"Block analysis expects CHW or BCHW input, got shape {tuple(feature.shape)}."
    )


def split_feature_blocks(
    feature: torch.Tensor,
    grid_size: int = 4,
) -> dict[tuple[int, int], torch.Tensor]:
    """Split a CHW/BCHW feature map into an equal spatial grid."""

    if not isinstance(grid_size, int) or grid_size <= 0:
        raise ValueError("grid_size must be a positive integer.")
    chw = _as_chw(feature)
    _, height, width = chw.shape
    if height % grid_size != 0 or width % grid_size != 0:
        raise ValueError(
            f"Feature spatial shape {(height, width)} is not divisible by "
            f"grid_size={grid_size}."
        )
    block_height = height // grid_size
    block_width = width // grid_size
    blocks: dict[tuple[int, int], torch.Tensor] = {}
    for row in range(grid_size):
        for col in range(grid_size):
            blocks[(row, col)] = chw[
                :,
                row * block_height : (row + 1) * block_height,
                col * block_width : (col + 1) * block_width,
            ]
    return blocks


def _pair_metrics(
    current: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[float, float]:
    if current.shape != reference.shape:
        raise ValueError(
            "Feature block shape mismatch: "
            f"current={tuple(current.shape)}, reference={tuple(reference.shape)}"
        )
    current = current.detach().float().reshape(-1)
    reference = reference.detach().float().reshape(-1)
    denominator = current.norm(p=2) * reference.norm(p=2)
    cosine = torch.dot(current, reference) / (denominator + eps)
    relative_l2 = (current - reference).norm(p=2) / (current.norm(p=2) + eps)
    return float(cosine.clamp(-1.0, 1.0).item()), float(relative_l2.item())


def block_metrics(
    current: torch.Tensor,
    reference: torch.Tensor,
    grid_size: int = 4,
) -> list[dict[str, float | int]]:
    """Return cosine and relative-L2 metrics for corresponding blocks."""

    if current.shape != reference.shape:
        raise ValueError(
            "Feature shape mismatch: "
            f"current={tuple(current.shape)}, reference={tuple(reference.shape)}"
        )
    current_blocks = split_feature_blocks(current, grid_size=grid_size)
    reference_blocks = split_feature_blocks(reference, grid_size=grid_size)
    rows: list[dict[str, float | int]] = []
    for row in range(grid_size):
        for col in range(grid_size):
            cosine, relative_l2 = _pair_metrics(
                current_blocks[(row, col)],
                reference_blocks[(row, col)],
            )
            rows.append(
                {
                    "block_row": row,
                    "block_col": col,
                    "cosine": cosine,
                    "relative_l2": relative_l2,
                }
            )
    return rows


class RollingFeatureBuffer:
    """A frame-indexed feature buffer with a strict maximum size."""

    def __init__(self, max_size: int) -> None:
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("max_size must be a positive integer.")
        self.max_size = max_size
        self._items: OrderedDict[int, list[torch.Tensor]] = OrderedDict()
        self.peak_size = 0

    def add(self, frame_id: int, features: Iterable[torch.Tensor]) -> None:
        copied = [feature.detach() for feature in features]
        self._items[int(frame_id)] = copied
        self._items.move_to_end(int(frame_id))
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)
        self.peak_size = max(self.peak_size, len(self._items))

    def get(self, frame_id: int) -> list[torch.Tensor] | None:
        return self._items.get(int(frame_id))

    def __len__(self) -> int:
        return len(self._items)

    def frame_ids(self) -> tuple[int, ...]:
        return tuple(self._items.keys())
