"""Controlled temporal reuse at hierarchical SegFormer stage boundaries.

The module intentionally contains only the three boundaries in the final
boundary study.  A skipped stage is never executed and masked afterwards;
the corresponding cached feature is passed directly to the decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


BOUNDARIES = ("s4", "s3_s4", "s2_s4")


@dataclass
class BoundaryCacheState:
    cached_f2: torch.Tensor | None = None
    cached_f3: torch.Tensor | None = None
    cached_f4: torch.Tensor | None = None
    initialized: bool = False
    age: int = 0

    @property
    def persistent_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.cached_f2, self.cached_f3, self.cached_f4)
            if tensor is not None
        )


class StructuredBoundaryReusePipeline:
    """Run one fixed reuse boundary with dense refreshes and cached skips."""

    def __init__(self, model, boundary: str = "s4") -> None:
        if boundary not in BOUNDARIES:
            raise ValueError(f"boundary must be one of {BOUNDARIES}")
        self.model = model
        self.boundary = boundary
        self.encoder = model.segformer.encoder
        self.call_counts = {f"stage{i}": 0 for i in range(1, 5)}
        self.call_counts["decoder"] = 0

    def reset_counts(self) -> None:
        for key in self.call_counts:
            self.call_counts[key] = 0

    def _run_stage(self, index: int, hidden: torch.Tensor) -> torch.Tensor:
        hidden, height, width = self.encoder.patch_embeddings[index](hidden)
        for block in self.encoder.block[index]:
            hidden = block(
                hidden,
                height,
                width,
                output_attentions=False,
            )[0]
        hidden = self.encoder.layer_norm[index](hidden)
        output = (
            hidden.reshape(hidden.shape[0], height, width, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        self.call_counts[f"stage{index + 1}"] += 1
        return output

    def _run_fresh_prefix(self, pixel_values: torch.Tensor, last_stage: int) -> list[torch.Tensor]:
        hidden = pixel_values
        features: list[torch.Tensor] = []
        for index in range(last_stage):
            hidden = self._run_stage(index, hidden)
            features.append(hidden)
        return features

    def _run_stage_range(self, hidden: torch.Tensor, start: int, stop: int) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        for index in range(start, stop):
            hidden = self._run_stage(index, hidden)
            features.append(hidden)
        return features

    def _cache_for(self, state: BoundaryCacheState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.cached_f2 is None or state.cached_f3 is None or state.cached_f4 is None:
            raise RuntimeError("initialized boundary cache is incomplete")
        return state.cached_f2, state.cached_f3, state.cached_f4

    def forward(
        self,
        pixel_values: torch.Tensor,
        refresh: bool,
        state: BoundaryCacheState | None = None,
    ) -> tuple[torch.Tensor, BoundaryCacheState, dict[str, object]]:
        if state is None:
            state = BoundaryCacheState()
        actual_refresh = bool(refresh) or not state.initialized

        if actual_refresh:
            features = self._run_fresh_prefix(pixel_values, 4)
            f1, f2, f3, f4 = features
            state.cached_f2 = f2.detach()
            state.cached_f3 = f3.detach()
            state.cached_f4 = f4.detach()
            state.initialized = True
            state.age = 0
        else:
            f1 = self._run_fresh_prefix(pixel_values, 1)[0]
            cached_f2, cached_f3, cached_f4 = self._cache_for(state)
            if self.boundary == "s4":
                f2, f3 = self._run_stage_range(f1, 1, 3)
                f4 = cached_f4
            elif self.boundary == "s3_s4":
                f2 = self._run_stage_range(f1, 1, 2)[0]
                f3, f4 = cached_f3, cached_f4
            else:
                f2, f3, f4 = cached_f2, cached_f3, cached_f4
            state.age += 1

        self.call_counts["decoder"] += 1
        logits = self.model.decode_head([f1, f2, f3, f4])
        return logits, state, {
            "boundary": self.boundary,
            "requested_refresh": bool(refresh),
            "actual_refresh": actual_refresh,
            "stage_calls": dict(self.call_counts),
            "persistent_cache_bytes": state.persistent_bytes,
            "cache_age": state.age,
        }

    def forward_segment(
        self,
        pixel_values: torch.Tensor,
        state: BoundaryCacheState | None = None,
    ) -> tuple[torch.Tensor, BoundaryCacheState, dict[str, object]]:
        """Process one contiguous same-video segment beginning at refresh.

        Prefix stages are evaluated for the complete segment.  The cached
        suffix is evaluated only for the first sample, then expanded for the
        remaining samples.  This preserves frame order while allowing the
        quality benchmark to use a small batch without executing skipped
        stages for every frame.
        """
        if state is None:
            state = BoundaryCacheState()
        if state.initialized:
            raise ValueError("forward_segment must start with a fresh segment state")
        batch_size = pixel_values.shape[0]
        if batch_size == 0:
            raise ValueError("segment must contain at least one frame")
        if self.boundary == "s4":
            f1, f2, f3 = self._run_fresh_prefix(pixel_values, 3)
            f4 = self._run_stage(3, f3[:1])
        elif self.boundary == "s3_s4":
            f1, f2 = self._run_fresh_prefix(pixel_values, 2)
            f3 = self._run_stage(2, f2[:1])
            f4 = self._run_stage(3, f3)
        else:
            f1 = self._run_fresh_prefix(pixel_values, 1)[0]
            f2 = self._run_stage(1, f1[:1])
            f3 = self._run_stage(2, f2)
            f4 = self._run_stage(3, f3)
        state.cached_f2 = f2[:1].detach()
        state.cached_f3 = f3[:1].detach()
        state.cached_f4 = f4[:1].detach()
        state.initialized = True
        state.age = 0
        if batch_size > 1:
            if self.boundary == "s3_s4":
                f3 = f3.expand(batch_size, -1, -1, -1)
            elif self.boundary == "s2_s4":
                f2 = f2.expand(batch_size, -1, -1, -1)
                f3 = f3.expand(batch_size, -1, -1, -1)
            f4 = f4.expand(batch_size, -1, -1, -1)
        self.call_counts["decoder"] += 1
        logits = self.model.decode_head([f1, f2, f3, f4])
        return logits, state, {
            "boundary": self.boundary,
            "requested_refresh": True,
            "actual_refresh": True,
            "segment_size": batch_size,
            "stage_calls": dict(self.call_counts),
            "persistent_cache_bytes": state.persistent_bytes,
            "cache_age": state.age,
        }
