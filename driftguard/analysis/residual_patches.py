"""Residual patch extraction with candidate-only interpolation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def selected_residual_patches(
    current_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    block_indices: np.ndarray,
    grid: int = 4,
    size: int = 16,
) -> np.ndarray:
    indices = np.asarray(block_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("block_indices must be one-dimensional")
    if (indices < 0).any() or (indices >= grid * grid).any():
        raise ValueError("block index outside spatial grid")
    if current_rgb.shape != reference_rgb.shape:
        raise ValueError("current and reference frames must have the same shape")
    if len(indices) == 0:
        return np.empty((0, 3, size, size), dtype=np.float32)

    residual = np.abs(
        current_rgb.astype(np.float32) / 255.0
        - reference_rgb.astype(np.float32) / 255.0
    )
    height, width = residual.shape[:2]
    crops = []
    for index in indices:
        row, col = divmod(int(index), grid)
        crop = residual[
            row * height // grid : (row + 1) * height // grid,
            col * width // grid : (col + 1) * width // grid,
        ]
        crops.append(torch.from_numpy(crop).permute(2, 0, 1))
    tensor = torch.stack(crops)
    resized = F.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.numpy().astype(np.float32)
