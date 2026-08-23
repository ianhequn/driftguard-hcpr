"""Sequential stale-feature propagation through the SegFormer encoder."""

from __future__ import annotations

from typing import Iterable

import torch


def _reshape_stage(
    hidden_states: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
    is_last: bool,
    reshape_last_stage: bool,
) -> torch.Tensor:
    if not is_last or reshape_last_stage:
        return (
            hidden_states.reshape(batch_size, height, width, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
    return hidden_states


def continue_encoder_from_stage(
    model,
    cached_stage: torch.Tensor,
    stage_index: int,
) -> list[torch.Tensor]:
    """Recompute later stages from a cached 0-based stage output.

    The returned list contains only stages after ``stage_index``. The cached
    stage itself is already available to the caller and is not recomputed.
    """

    if stage_index < 0 or stage_index > 3:
        raise ValueError("stage_index must be between 0 and 3")
    encoder = model.segformer.encoder
    hidden_states = cached_stage
    batch_size = hidden_states.shape[0]
    outputs: list[torch.Tensor] = []

    for idx in range(stage_index + 1, len(encoder.patch_embeddings)):
        embedding_layer = encoder.patch_embeddings[idx]
        block_layer = encoder.block[idx]
        norm_layer = encoder.layer_norm[idx]
        hidden_states, height, width = embedding_layer(hidden_states)
        for block in block_layer:
            hidden_states = block(
                hidden_states,
                height,
                width,
                output_attentions=False,
            )[0]
        hidden_states = norm_layer(hidden_states)
        hidden_states = _reshape_stage(
            hidden_states,
            batch_size=batch_size,
            height=height,
            width=width,
            is_last=idx == len(encoder.patch_embeddings) - 1,
            reshape_last_stage=bool(encoder.config.reshape_last_stage),
        )
        outputs.append(hidden_states)
    return outputs


def sequential_stale_features(
    model,
    current_features: list[torch.Tensor],
    cached_features: list[torch.Tensor],
    cut: int,
) -> list[torch.Tensor]:
    """Build features for a cache cut using stale prefix and fresh suffix.

    ``cut`` is one-based: cut 1 reuses Stage 1, cut 4 reuses Stage 1-4 and
    only reruns the decoder.
    """

    if len(current_features) != 4 or len(cached_features) != 4:
        raise ValueError("Both feature lists must contain four stages")
    if cut < 1 or cut > 4:
        raise ValueError("cut must be between 1 and 4")

    result = list(current_features)
    result[cut - 1] = cached_features[cut - 1]
    if cut == 4:
        return result
    fresh_suffix = continue_encoder_from_stage(
        model,
        cached_features[cut - 1],
        stage_index=cut - 1,
    )
    result[cut:] = fresh_suffix
    return result


def decode_features(model, features: Iterable[torch.Tensor]) -> torch.Tensor:
    """Decode four NCHW stage features into native SegFormer logits."""

    values = list(features)
    if len(values) != 4:
        raise ValueError("Expected four stage features")
    return model.decode_head(values)
