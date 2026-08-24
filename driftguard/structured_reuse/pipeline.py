"""Fresh F1-F3 plus full-refresh-or-reuse F4 execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Stage4CacheState:
    """The only persistent model cache used by Round 6."""

    cached_f4: torch.Tensor | None = None
    age: int = 0
    initialized: bool = False

    @property
    def persistent_bytes(self) -> int:
        if self.cached_f4 is None:
            return 0
        return self.cached_f4.numel() * self.cached_f4.element_size()


class WholeStage4Cache:
    """Run the complete dense Stage 4 or skip it and reuse final F4."""

    def __init__(self, model):
        self.model = model
        self.encoder = model.segformer.encoder
        self.stage4_forward_calls = 0

    def reset_counter(self) -> None:
        self.stage4_forward_calls = 0

    def run_dense(self, f3: torch.Tensor) -> torch.Tensor:
        hidden, height, width = self.encoder.patch_embeddings[3](f3)
        for block in self.encoder.block[3]:
            hidden = block(
                hidden,
                height,
                width,
                output_attentions=False,
            )[0]
        hidden = self.encoder.layer_norm[3](hidden)
        self.stage4_forward_calls += 1
        return (
            hidden.reshape(hidden.shape[0], height, width, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def forward(
        self,
        f3: torch.Tensor,
        refresh: bool,
        state: Stage4CacheState | None,
    ) -> tuple[torch.Tensor, Stage4CacheState, bool]:
        if state is None:
            state = Stage4CacheState()
        actual_refresh = bool(refresh) or not state.initialized
        if actual_refresh:
            f4 = self.run_dense(f3)
            state.cached_f4 = f4.detach()
            state.age = 0
            state.initialized = True
            return f4, state, True
        if state.cached_f4 is None:
            raise RuntimeError("initialized Stage-4 cache has no cached F4")
        state.age += 1
        return state.cached_f4, state, False


class StructuredReusePipeline:
    """Execute fresh F1-F3, optional dense F4, and the full decoder."""

    def __init__(self, model):
        self.model = model
        self.encoder = model.segformer.encoder
        self.stage4_cache = WholeStage4Cache(model)

    def fresh_stage_1_to_3(
        self,
        pixel_values: torch.Tensor,
    ) -> list[torch.Tensor]:
        hidden = pixel_values
        features = []
        for index in range(3):
            hidden, height, width = self.encoder.patch_embeddings[index](
                hidden
            )
            for block in self.encoder.block[index]:
                hidden = block(
                    hidden,
                    height,
                    width,
                    output_attentions=False,
                )[0]
            hidden = self.encoder.layer_norm[index](hidden)
            hidden = (
                hidden.reshape(hidden.shape[0], height, width, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            features.append(hidden)
        return features

    def forward(
        self,
        pixel_values: torch.Tensor,
        refresh: bool,
        state: Stage4CacheState | None,
    ) -> tuple[torch.Tensor, Stage4CacheState, dict]:
        f1, f2, f3 = self.fresh_stage_1_to_3(pixel_values)
        calls_before = self.stage4_cache.stage4_forward_calls
        f4, state, actual_refresh = self.stage4_cache.forward(
            f3,
            refresh,
            state,
        )
        logits = self.model.decode_head([f1, f2, f3, f4])
        calls_after = self.stage4_cache.stage4_forward_calls
        return logits, state, {
            "requested_refresh": bool(refresh),
            "actual_refresh": actual_refresh,
            "stage4_calls_delta": calls_after - calls_before,
            "stage4_forward_calls": calls_after,
            "cache_age": state.age,
            "persistent_cache_bytes": state.persistent_bytes,
        }
