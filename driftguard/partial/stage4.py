"""Cached-context approximate partial execution for SegFormer Stage 4."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PartialComputeStats:
    """Token and cache traffic counters for one partial Stage-4 call."""

    logical_refresh_blocks: int
    logical_tokens: int
    physical_tokens: int
    full_tokens: int
    physical_compute_ratio: float
    halo_tokens: int
    cache_reads: int
    cache_writes: int
    full_refresh: bool


@dataclass
class PartialStage4Cache:
    """Persistent per-layer state used by cached-context execution."""

    patch_tokens: torch.Tensor
    layer_k: list[torch.Tensor]
    layer_v: list[torch.Tensor]
    layer_ffn_dense1: list[torch.Tensor]
    layer_outputs: list[torch.Tensor]
    output_tokens: torch.Tensor
    height: int
    width: int

    @property
    def persistent_bytes(self) -> int:
        tensors = [
            self.patch_tokens,
            self.output_tokens,
            *self.layer_k,
            *self.layer_v,
            *self.layer_ffn_dense1,
            *self.layer_outputs,
        ]
        return sum(value.numel() * value.element_size() for value in tensors)


class PartialStage4Engine:
    """Execute only refreshed Stage-4 tokens with stale cached context.

    Selected tokens use current patch embeddings, Q/K/V, attention, and FFN
    computation. Non-selected tokens retain historical outputs. Their cached
    K/V and MixFFN dense1 values remain available as stale global and local
    context for selected tokens.
    """

    def __init__(self, model, grid: int = 4):
        self.model = model
        self.encoder = model.segformer.encoder
        self.embedding = self.encoder.patch_embeddings[3]
        self.blocks = self.encoder.block[3]
        self.final_norm = self.encoder.layer_norm[3]
        self.grid = int(grid)
        if self.grid <= 0:
            raise ValueError("grid must be positive")
        self._block_token_cache: dict[
            tuple[int, int, str], torch.Tensor
        ] = {}

    @staticmethod
    def _flatten_spatial(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return (
            tokens.reshape(tokens.shape[0], height, width, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def _token_indices(
        self,
        refresh_mask: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, int]:
        mask = torch.as_tensor(refresh_mask, dtype=torch.bool, device="cpu")
        if mask.numel() != self.grid * self.grid:
            raise ValueError(
                f"refresh_mask must contain {self.grid * self.grid} blocks"
            )
        mask = mask.reshape(self.grid, self.grid)
        selected_blocks = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
        if height % self.grid or width % self.grid:
            raise ValueError("Stage-4 spatial shape must be divisible by grid")
        block_h = height // self.grid
        block_w = width // self.grid
        if selected_blocks.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device), 0
        cache_key = (height, width, str(device))
        block_tokens = self._block_token_cache.get(cache_key)
        if block_tokens is None:
            values = []
            for block_index in range(self.grid * self.grid):
                block_row, block_col = divmod(block_index, self.grid)
                rows = torch.arange(
                    block_row * block_h,
                    (block_row + 1) * block_h,
                    device=device,
                )
                cols = torch.arange(
                    block_col * block_w,
                    (block_col + 1) * block_w,
                    device=device,
                )
                values.append(
                    (rows[:, None] * width + cols[None, :]).reshape(-1)
                )
            block_tokens = torch.stack(values).long()
            self._block_token_cache[cache_key] = block_tokens
        indices = block_tokens.index_select(
            0, selected_blocks.to(device=device)
        ).reshape(-1)
        return indices.sort().values, int(selected_blocks.numel())

    @staticmethod
    def _project_kv(attention_self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = attention_self.transpose_for_scores(attention_self.key(normalized))
        value = attention_self.transpose_for_scores(attention_self.value(normalized))
        return key, value

    def _full_stage4_with_cache(
        self,
        stage3_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, PartialStage4Cache]:
        hidden_states, height, width = self.embedding(stage3_feature)
        patch_tokens = hidden_states.detach().clone()
        layer_k = []
        layer_v = []
        layer_ffn_dense1 = []
        layer_outputs = []

        for block in self.blocks:
            normalized_1 = block.layer_norm_1(hidden_states)
            key, value = self._project_kv(block.attention.self, normalized_1)
            attention_output = block.attention(
                normalized_1,
                height,
                width,
                output_attentions=False,
            )[0]
            attention_hidden = block.drop_path(attention_output) + hidden_states
            normalized_2 = block.layer_norm_2(attention_hidden)
            dense1 = block.mlp.dense1(normalized_2)
            mlp_output = block.mlp.dwconv(dense1, height, width)
            mlp_output = block.mlp.intermediate_act_fn(mlp_output)
            mlp_output = block.mlp.dropout(mlp_output)
            mlp_output = block.mlp.dense2(mlp_output)
            mlp_output = block.mlp.dropout(mlp_output)
            hidden_states = block.drop_path(mlp_output) + attention_hidden

            layer_k.append(key.detach().clone())
            layer_v.append(value.detach().clone())
            layer_ffn_dense1.append(dense1.detach().clone())
            layer_outputs.append(hidden_states.detach().clone())

        output_tokens = self.final_norm(hidden_states)
        output = self._flatten_spatial(output_tokens, height, width)
        cache = PartialStage4Cache(
            patch_tokens=patch_tokens,
            layer_k=layer_k,
            layer_v=layer_v,
            layer_ffn_dense1=layer_ffn_dense1,
            layer_outputs=layer_outputs,
            output_tokens=output_tokens.detach().clone(),
            height=height,
            width=width,
        )
        return output, cache

    @staticmethod
    def _sparse_conv2d_tokens(
        feature: torch.Tensor,
        conv: torch.nn.Conv2d,
        indices: torch.Tensor,
        output_height: int,
        output_width: int,
    ) -> torch.Tensor:
        if conv.kernel_size != (3, 3) or conv.padding != (1, 1):
            raise ValueError("Sparse patch embedding requires Conv2d 3x3 padding 1")
        if conv.stride != (2, 2) or conv.dilation != (1, 1):
            raise ValueError("Sparse patch embedding requires stride 2, dilation 1")
        batch, channels, _, _ = feature.shape
        padded = F.pad(feature, (1, 1, 1, 1))
        padded_height, padded_width = padded.shape[-2:]
        rows = torch.div(indices, output_width, rounding_mode="floor")
        cols = indices.remainder(output_width)
        offsets = torch.arange(3, device=feature.device)
        patch_rows = rows[:, None, None] * 2 + offsets[None, :, None]
        patch_cols = cols[:, None, None] * 2 + offsets[None, None, :]
        spatial_indices = (
            patch_rows * padded_width + patch_cols
        ).reshape(-1)
        flat = padded.permute(0, 2, 3, 1).reshape(
            batch, padded_height * padded_width, channels
        )
        patches = flat.index_select(1, spatial_indices).reshape(
            batch, indices.numel(), 9, channels
        )
        patches = patches.permute(0, 1, 3, 2).reshape(
            batch, indices.numel(), channels * 9
        )
        weight = conv.weight.reshape(conv.out_channels, channels * 9)
        output = torch.matmul(patches, weight.t())
        if conv.bias is not None:
            output = output + conv.bias
        return output

    @staticmethod
    def _sparse_depthwise(
        dense1_cache: torch.Tensor,
        conv: torch.nn.Conv2d,
        indices: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch, _, channels = dense1_cache.shape
        spatial = (
            dense1_cache.reshape(batch, height, width, channels)
            .permute(0, 3, 1, 2)
        )
        padded = F.pad(spatial, (1, 1, 1, 1))
        padded_width = width + 2
        rows = torch.div(indices, width, rounding_mode="floor")
        cols = indices.remainder(width)
        offsets = torch.arange(3, device=indices.device)
        patch_rows = rows[:, None, None] + offsets[None, :, None]
        patch_cols = cols[:, None, None] + offsets[None, None, :]
        spatial_indices = (
            patch_rows * padded_width + patch_cols
        ).reshape(-1)
        flat = padded.permute(0, 2, 3, 1).reshape(
            batch, (height + 2) * (width + 2), channels
        )
        patches = flat.index_select(1, spatial_indices).reshape(
            batch, indices.numel(), 9, channels
        )
        patches = patches.permute(0, 1, 3, 2)
        weight = conv.weight.reshape(channels, 9)
        output = (patches * weight[None, None, :, :]).sum(dim=-1)
        if conv.bias is not None:
            output = output + conv.bias
        return output

    @staticmethod
    def _selected_attention(
        block,
        normalized_selected: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        attention_self = block.attention.self
        query = attention_self.transpose_for_scores(
            attention_self.query(normalized_selected)
        )
        scores = torch.matmul(query, key_cache.transpose(-1, -2))
        scores = scores / math.sqrt(attention_self.attention_head_size)
        probabilities = F.softmax(scores, dim=-1)
        probabilities = attention_self.dropout(probabilities)
        context = torch.matmul(probabilities, value_cache)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.reshape(
            context.shape[0],
            context.shape[1],
            attention_self.all_head_size,
        )
        return block.attention.output(context, normalized_selected)

    def initialize(
        self,
        stage3_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, PartialStage4Cache, PartialComputeStats]:
        output, cache = self._full_stage4_with_cache(stage3_feature)
        full_tokens = cache.height * cache.width
        stats = PartialComputeStats(
            logical_refresh_blocks=self.grid * self.grid,
            logical_tokens=full_tokens,
            physical_tokens=full_tokens,
            full_tokens=full_tokens,
            physical_compute_ratio=1.0,
            halo_tokens=0,
            cache_reads=0,
            cache_writes=full_tokens,
            full_refresh=True,
        )
        return output, cache, stats

    def partial_stage4(
        self,
        stage3_feature: torch.Tensor,
        cache_state: PartialStage4Cache | None,
        refresh_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, PartialStage4Cache, PartialComputeStats]:
        """Return Stage-4 output, updated cache, and physical compute counters."""

        if cache_state is None:
            return self.initialize(stage3_feature)
        height, width = cache_state.height, cache_state.width
        full_tokens = height * width
        indices, selected_blocks = self._token_indices(
            refresh_mask,
            height,
            width,
            stage3_feature.device,
        )
        selected_tokens = int(indices.numel())
        if selected_tokens == full_tokens:
            return self.initialize(stage3_feature)
        if selected_tokens == 0:
            stats = PartialComputeStats(
                logical_refresh_blocks=0,
                logical_tokens=0,
                physical_tokens=0,
                full_tokens=full_tokens,
                physical_compute_ratio=0.0,
                halo_tokens=0,
                cache_reads=full_tokens,
                cache_writes=0,
                full_refresh=False,
            )
            output = self._flatten_spatial(
                cache_state.output_tokens,
                height,
                width,
            )
            return output, cache_state, stats

        patch_selected = self._sparse_conv2d_tokens(
            stage3_feature,
            self.embedding.proj,
            indices,
            height,
            width,
        )
        patch_selected = self.embedding.layer_norm(patch_selected)
        cache_state.patch_tokens.index_copy_(1, indices, patch_selected)
        hidden_selected = patch_selected

        for layer_index, block in enumerate(self.blocks):
            normalized_1 = block.layer_norm_1(hidden_selected)
            key_selected, value_selected = self._project_kv(
                block.attention.self,
                normalized_1,
            )
            cache_state.layer_k[layer_index].index_copy_(
                2, indices, key_selected
            )
            cache_state.layer_v[layer_index].index_copy_(
                2, indices, value_selected
            )
            attention_output = self._selected_attention(
                block,
                normalized_1,
                cache_state.layer_k[layer_index],
                cache_state.layer_v[layer_index],
            )
            attention_hidden = (
                block.drop_path(attention_output) + hidden_selected
            )
            normalized_2 = block.layer_norm_2(attention_hidden)
            dense1_selected = block.mlp.dense1(normalized_2)
            cache_state.layer_ffn_dense1[layer_index].index_copy_(
                1, indices, dense1_selected
            )
            mlp_output = self._sparse_depthwise(
                cache_state.layer_ffn_dense1[layer_index],
                block.mlp.dwconv.dwconv,
                indices,
                height,
                width,
            )
            mlp_output = block.mlp.intermediate_act_fn(mlp_output)
            mlp_output = block.mlp.dropout(mlp_output)
            mlp_output = block.mlp.dense2(mlp_output)
            mlp_output = block.mlp.dropout(mlp_output)
            hidden_selected = (
                block.drop_path(mlp_output) + attention_hidden
            )
            cache_state.layer_outputs[layer_index].index_copy_(
                1, indices, hidden_selected
            )

        output_selected = self.final_norm(hidden_selected)
        cache_state.output_tokens.index_copy_(1, indices, output_selected)
        output = self._flatten_spatial(
            cache_state.output_tokens,
            height,
            width,
        )
        layer_count = len(self.blocks)
        stats = PartialComputeStats(
            logical_refresh_blocks=selected_blocks,
            logical_tokens=selected_tokens,
            physical_tokens=selected_tokens,
            full_tokens=full_tokens,
            physical_compute_ratio=selected_tokens / full_tokens,
            halo_tokens=0,
            cache_reads=layer_count * (
                2 * full_tokens + 9 * selected_tokens
            ),
            cache_writes=selected_tokens * (2 + 5 * layer_count),
            full_refresh=False,
        )
        return output, cache_state, stats
