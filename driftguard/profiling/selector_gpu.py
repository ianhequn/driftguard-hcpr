"""GPU-resident preprocessing for the frozen AFVT selector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from driftguard.analysis.age_aware_schedulers import (
    SchedulerDecision,
    partition_afvt,
)


@dataclass
class GPUSelectorResult:
    decision: SchedulerDecision
    candidate_indices: np.ndarray
    risk_scores: np.ndarray
    used_visual_path: bool


def _validate_frame_pair(
    current: torch.Tensor,
    reference: torch.Tensor,
    grid: int,
) -> None:
    if current.shape != reference.shape:
        raise ValueError("current and reference frames must have the same shape")
    if current.ndim != 4 or current.shape[0] != 1 or current.shape[1] != 3:
        raise ValueError("frames must have shape [1, 3, H, W]")
    if current.device != reference.device:
        raise ValueError("current and reference frames must share a device")
    if current.shape[-2] % grid or current.shape[-1] % grid:
        raise ValueError("frame height and width must be divisible by grid")


def gpu_rgb_residual(
    current: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Reproduce Round 3 RGB residual semantics on the tensor device."""

    if current.dtype == torch.uint8:
        current_float = current.float().div(255.0)
        reference_float = reference.float().div(255.0)
    elif current.is_floating_point():
        current_float = current.float()
        reference_float = reference.float()
    else:
        raise TypeError("frames must be uint8 or floating point tensors")
    return torch.abs(current_float - reference_float)


def vectorized_all_blocks(
    residual: torch.Tensor,
    grid: int = 4,
) -> torch.Tensor:
    """Partition BCHW residual into B*grid*grid block tensors."""

    if residual.ndim != 4:
        raise ValueError("residual must have shape [B, C, H, W]")
    batch, channels, height, width = residual.shape
    if height % grid or width % grid:
        raise ValueError("residual height and width must be divisible by grid")
    block_h = height // grid
    block_w = width // grid
    return (
        residual.view(batch, channels, grid, block_h, grid, block_w)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch * grid * grid, channels, block_h, block_w)
    )


def resize_candidate_blocks(
    blocks: torch.Tensor,
    size: int = 16,
) -> torch.Tensor:
    if blocks.ndim != 4:
        raise ValueError("blocks must have shape [N, C, H, W]")
    if blocks.shape[0] == 0:
        return blocks.new_empty((0, blocks.shape[1], size, size))
    return F.interpolate(
        blocks,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )


class GPUAFVTSelector:
    """Frozen AFVT with CUDA-resident image preprocessing and batched CNN."""

    def __init__(
        self,
        model: torch.nn.Module,
        metadata_mean: np.ndarray,
        metadata_std: np.ndarray,
        *,
        grid: int = 4,
        patch_size: int = 16,
        extraction: str = "direct",
        fast_path: bool = True,
    ):
        if extraction not in {"all_blocks", "direct"}:
            raise ValueError("extraction must be all_blocks or direct")
        self.model = model.eval()
        self.grid = grid
        self.patch_size = patch_size
        self.extraction = extraction
        self.fast_path = fast_path
        self.metadata_mean = np.asarray(metadata_mean, dtype=np.float32)
        self.metadata_std = np.asarray(metadata_std, dtype=np.float32)
        if self.metadata_mean.shape != (1,) or self.metadata_std.shape != (1,):
            raise ValueError("frozen Patch+Age predictor requires one metadata value")
        if self.metadata_std[0] < 1e-8:
            self.metadata_std[0] = 1.0
        self._linear_index_cache: dict[
            tuple[int, int, str], torch.Tensor
        ] = {}

    def _block_linear_indices(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = (height, width, str(device))
        cached = self._linear_index_cache.get(key)
        if cached is not None:
            return cached
        block_h = height // self.grid
        block_w = width // self.grid
        rows = []
        for index in range(self.grid * self.grid):
            row, col = divmod(index, self.grid)
            y = torch.arange(
                row * block_h,
                (row + 1) * block_h,
                device=device,
                dtype=torch.int64,
            )
            x = torch.arange(
                col * block_w,
                (col + 1) * block_w,
                device=device,
                dtype=torch.int64,
            )
            rows.append((y[:, None] * width + x[None, :]).reshape(-1))
        cached = torch.stack(rows)
        self._linear_index_cache[key] = cached
        return cached

    def _direct_candidate_blocks(
        self,
        current: torch.Tensor,
        reference: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        _, channels, height, width = current.shape
        block_h = height // self.grid
        block_w = width // self.grid
        linear = self._block_linear_indices(
            height,
            width,
            current.device,
        ).index_select(0, indices)
        selected = linear.reshape(-1)
        current_values = current[0].reshape(channels, -1).index_select(
            1, selected
        )
        reference_values = reference[0].reshape(channels, -1).index_select(
            1, selected
        )
        current_blocks = (
            current_values.view(channels, len(indices), block_h, block_w)
            .permute(1, 0, 2, 3)
        )
        reference_blocks = (
            reference_values.view(channels, len(indices), block_h, block_w)
            .permute(1, 0, 2, 3)
        )
        return gpu_rgb_residual(current_blocks, reference_blocks)

    def candidate_patches(
        self,
        current: torch.Tensor,
        reference: torch.Tensor,
        candidate_indices: np.ndarray,
    ) -> torch.Tensor:
        _validate_frame_pair(current, reference, self.grid)
        indices = torch.as_tensor(
            candidate_indices,
            dtype=torch.int64,
            device=current.device,
        )
        if self.extraction == "all_blocks":
            residual = gpu_rgb_residual(current, reference)
            blocks = vectorized_all_blocks(residual, self.grid).index_select(
                0, indices
            )
        else:
            blocks = self._direct_candidate_blocks(
                current,
                reference,
                indices,
            )
        return resize_candidate_blocks(blocks, self.patch_size)

    def score_candidates(
        self,
        current: torch.Tensor,
        reference: torch.Tensor,
        ages: np.ndarray,
        candidate_indices: np.ndarray,
    ) -> np.ndarray:
        patches = self.candidate_patches(
            current,
            reference,
            candidate_indices,
        )
        normalized = (
            (
                np.asarray(ages, dtype=np.float32)[candidate_indices, None]
                - self.metadata_mean
            )
            / self.metadata_std
        ).astype(np.float32)
        metadata = torch.from_numpy(normalized).to(current.device)
        with torch.inference_mode():
            scores = self.model(patches, metadata)
        return scores.detach().cpu().numpy()

    def select(
        self,
        current: torch.Tensor,
        reference: torch.Tensor,
        ages: np.ndarray,
        ratio: float,
    ) -> GPUSelectorResult:
        values = np.asarray(ages, dtype=np.float32)
        partition = partition_afvt(values, ratio)
        mask = np.zeros(len(values), dtype=bool)
        mask[partition.older] = True

        if not partition.needs_visual_ranking and self.fast_path:
            if partition.remaining:
                mask[partition.tie[: partition.remaining]] = True
            decision = SchedulerDecision(
                mask=mask,
                cnn_indices=np.empty(0, dtype=np.int64),
                tie_group_size=int(len(partition.tie)),
                forced_blocks=int(len(partition.older)),
                visual_selected_blocks=int(partition.remaining),
                forced_overflow=False,
                max_age=None,
            )
            return GPUSelectorResult(
                decision=decision,
                candidate_indices=np.empty(0, dtype=np.int64),
                risk_scores=np.empty(0, dtype=np.float32),
                used_visual_path=False,
            )

        candidates = partition.tie.astype(np.int64, copy=False)
        scores = self.score_candidates(
            current,
            reference,
            values,
            candidates,
        )
        order = np.lexsort((candidates, -scores.astype(np.float64)))
        selected = candidates[order[: partition.remaining]]
        mask[selected] = True
        decision = SchedulerDecision(
            mask=mask,
            cnn_indices=candidates.copy(),
            tie_group_size=int(len(candidates)),
            forced_blocks=int(len(partition.older)),
            visual_selected_blocks=int(len(selected)),
            forced_overflow=False,
            max_age=None,
        )
        return GPUSelectorResult(
            decision=decision,
            candidate_indices=candidates.copy(),
            risk_scores=scores.astype(np.float32, copy=False),
            used_visual_path=True,
        )
