"""Cheap frame/block observables used by the drift dataset and policies."""

from __future__ import annotations

import numpy as np
import torch


def _grid_slices(height: int, width: int, grid: int):
    if height % grid != 0 or width % grid != 0:
        raise ValueError(f"Spatial shape {(height, width)} is not divisible by grid {grid}")
    block_height = height // grid
    block_width = width // grid
    for row in range(grid):
        for col in range(grid):
            yield (
                row,
                col,
                slice(row * block_height, (row + 1) * block_height),
                slice(col * block_width, (col + 1) * block_width),
            )


def rgb_block_observables(
    current_rgb: np.ndarray,
    previous_rgb: np.ndarray | None,
    grid: int = 4,
) -> np.ndarray:
    """Return local residual, global residual, and local variance per block."""

    current = current_rgb.astype(np.float32) / 255.0
    if previous_rgb is None:
        residual = np.zeros_like(current, dtype=np.float32)
    else:
        previous = previous_rgb.astype(np.float32) / 255.0
        if previous.shape != current.shape:
            raise ValueError("Current and previous frames must have the same shape")
        residual = np.abs(current - previous).mean(axis=2)
    global_residual = float(residual.mean())
    values = []
    for _, _, row_slice, col_slice in _grid_slices(
        residual.shape[0],
        residual.shape[1],
        grid,
    ):
        block = residual[row_slice, col_slice]
        values.append(
            [
                float(block.mean()),
                0.0,
                global_residual,
                float(block.var()),
            ]
        )
    return np.asarray(values, dtype=np.float32)


def entropy_block_observables(
    logits: torch.Tensor | None,
    image_height: int,
    image_width: int,
    grid: int = 4,
) -> np.ndarray:
    """Return normalized prediction entropy averaged over image blocks."""

    if logits is None:
        return np.zeros((grid * grid,), dtype=np.float32)
    values = logits.detach().float()
    if values.ndim != 4:
        raise ValueError("Logits must be BCHW")
    probabilities = torch.softmax(values, dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)
    entropy = torch.nn.functional.interpolate(
        entropy.unsqueeze(1),
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    entropy = entropy / np.log(max(2, values.shape[1]))
    array = entropy.cpu().numpy()
    output = []
    for _, _, row_slice, col_slice in _grid_slices(
        image_height,
        image_width,
        grid,
    ):
        output.append(float(array[row_slice, col_slice].mean()))
    return np.asarray(output, dtype=np.float32)


def block_relative_l2(
    current: torch.Tensor,
    reference: torch.Tensor,
    grid: int = 4,
    eps: float = 1e-8,
) -> np.ndarray:
    """Return relative L2 drift for each spatial feature block."""

    if current.shape != reference.shape:
        raise ValueError("Feature shapes must match")
    if current.ndim != 4 or current.shape[0] != 1:
        raise ValueError("Features must have shape [1,C,H,W]")
    output = []
    for _, _, row_slice, col_slice in _grid_slices(
        current.shape[2],
        current.shape[3],
        grid,
    ):
        current_block = current[:, :, row_slice, col_slice].float()
        reference_block = reference[:, :, row_slice, col_slice].float()
        value = (current_block - reference_block).norm() / (
            current_block.norm() + eps
        )
        output.append(float(value.item()))
    return np.asarray(output, dtype=np.float32)


def replace_blocks(
    cache: torch.Tensor,
    current: torch.Tensor,
    refresh_mask: np.ndarray,
    grid: int = 4,
) -> torch.Tensor:
    """Return a cache tensor with selected spatial blocks replaced."""

    if cache.shape != current.shape:
        raise ValueError("Cache and current feature shapes must match")
    if refresh_mask.shape != (grid * grid,):
        raise ValueError("Refresh mask has the wrong number of blocks")
    updated = cache.clone()
    for index, (_, _, row_slice, col_slice) in enumerate(
        _grid_slices(current.shape[2], current.shape[3], grid)
    ):
        if bool(refresh_mask[index]):
            updated[:, :, row_slice, col_slice] = current[:, :, row_slice, col_slice]
    return updated
