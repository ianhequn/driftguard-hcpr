"""Hardware-calibrated progressive whole-stage reuse for Round 11A.

The controller computes F1 on every frame and progressively decides whether
to continue to stages 2, 3, and 4.  A stopped suffix is physically absent from
that frame; its cached features are passed directly to the decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


ROUTES = ("S2-S4", "S3-S4", "S4", "Full")
FORCED_ROUTES = ("s2_s4", "s3_s4", "s4", "full")


@dataclass
class HCPRCacheState:
    """Per-video HCPR state.  References are pooled FP32 tensors."""

    cached_f2: torch.Tensor | None = None
    cached_f3: torch.Tensor | None = None
    cached_f4: torch.Tensor | None = None
    reference_1_to_2: torch.Tensor | None = None
    reference_2_to_3: torch.Tensor | None = None
    reference_3_to_4: torch.Tensor | None = None
    age2: int = 0
    age3: int = 0
    age4: int = 0
    initialized: bool = False
    max_stale_run: int | None = 7

    @property
    def persistent_bytes(self) -> int:
        tensors = (
            self.cached_f2,
            self.cached_f3,
            self.cached_f4,
            self.reference_1_to_2,
            self.reference_2_to_3,
            self.reference_3_to_4,
        )
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)


class HardwareCalibratedProgressiveReusePipeline:
    """Progressively route a frame through fresh encoder stages.

    ``stage_costs`` contains normalized marginal costs ``k2``, ``k3``, and
    ``k4``.  ``drift_scales`` contains the unlabeled calibration percentiles
    for the three stage-boundary drift signals.
    """

    def __init__(
        self,
        model: Any,
        stage_costs: dict[int | str, float] | None = None,
        drift_scales: dict[int | str, float] | None = None,
        lambda_value: float = 1.0,
        max_stale_run: int | None = 7,
    ) -> None:
        if lambda_value < 0:
            raise ValueError("lambda_value must be non-negative")
        if max_stale_run is not None and max_stale_run < 0:
            raise ValueError("max_stale_run must be non-negative or None")
        self.model = model
        self.encoder = model.segformer.encoder
        self.lambda_value = float(lambda_value)
        self.max_stale_run = max_stale_run
        self.stage_costs = self._normalize_parameters(stage_costs, "k", default=1.0)
        self.drift_scales = self._normalize_parameters(drift_scales, "scale", default=1.0)
        self.call_counts = {f"stage{i}": 0 for i in range(1, 5)}
        self.call_counts["decoder"] = 0

    @staticmethod
    def _normalize_parameters(values: dict[int | str, float] | None, prefix: str, default: float) -> dict[int, float]:
        result: dict[int, float] = {}
        for stage in (2, 3, 4):
            candidates = (stage, f"{prefix}{stage}")
            value = next((values[key] for key in candidates if values is not None and key in values), default)
            value = float(value)
            if value <= 0:
                raise ValueError(f"{prefix}{stage} must be positive")
            result[stage] = value
        return result

    def reset_counts(self) -> None:
        for key in self.call_counts:
            self.call_counts[key] = 0

    @staticmethod
    def new_state(max_stale_run: int | None = 7) -> HCPRCacheState:
        return HCPRCacheState(max_stale_run=max_stale_run)

    def _run_stage(self, index: int, hidden: torch.Tensor) -> torch.Tensor:
        hidden, height, width = self.encoder.patch_embeddings[index](hidden)
        for block in self.encoder.block[index]:
            hidden = block(hidden, height, width, output_attentions=False)[0]
        hidden = self.encoder.layer_norm[index](hidden)
        self.call_counts[f"stage{index + 1}"] += 1
        return hidden.reshape(hidden.shape[0], height, width, -1).permute(0, 3, 1, 2).contiguous()

    def _pool(self, feature: torch.Tensor) -> torch.Tensor:
        # The reduction and cosine calculation stay on the GPU in FP32.
        return F.adaptive_avg_pool2d(feature.float(), output_size=1).flatten(1)

    def _drift(self, current: torch.Tensor, reference: torch.Tensor | None) -> torch.Tensor:
        if reference is None:
            return current.new_zeros((), dtype=torch.float32)
        current_pooled = self._pool(current)
        current_unit = F.normalize(current_pooled, dim=1, eps=1e-12)
        reference_unit = F.normalize(reference.float(), dim=1, eps=1e-12)
        return (1.0 - (current_unit * reference_unit).sum(dim=1)).mean()

    def _score(self, drift: torch.Tensor, stage: int) -> torch.Tensor:
        scale = self.drift_scales[stage]
        cost = self.stage_costs[stage]
        return drift / (scale + 1e-12) / cost

    def _guard_forces(self, age: int) -> bool:
        return self.max_stale_run is not None and age >= self.max_stale_run

    def _cache_ready(self, state: HCPRCacheState) -> None:
        if not state.initialized or any(
            value is None
            for value in (
                state.cached_f2,
                state.cached_f3,
                state.cached_f4,
                state.reference_1_to_2,
                state.reference_2_to_3,
                state.reference_3_to_4,
            )
        ):
            raise RuntimeError("HCPR cache is not initialized")

    def _increment_reuse_ages(self, state: HCPRCacheState, route: str) -> None:
        if route == "S3-S4":
            state.age3 += 1
            state.age4 += 1
        elif route == "S4":
            state.age4 += 1
        elif route == "S2-S4":
            state.age2 += 1
            state.age3 += 1
            state.age4 += 1
        else:
            raise ValueError(f"unsupported reuse route {route}")

    def _full_frame(self, pixel_values: torch.Tensor, state: HCPRCacheState) -> list[torch.Tensor]:
        f1 = self._run_stage(0, pixel_values)
        f2 = self._run_stage(1, f1)
        f3 = self._run_stage(2, f2)
        f4 = self._run_stage(3, f3)
        state.cached_f2 = f2.detach()
        state.cached_f3 = f3.detach()
        state.cached_f4 = f4.detach()
        state.reference_1_to_2 = self._pool(f1).detach()
        state.reference_2_to_3 = self._pool(f2).detach()
        state.reference_3_to_4 = self._pool(f3).detach()
        state.age2 = state.age3 = state.age4 = 0
        state.initialized = True
        return [f1, f2, f3, f4]

    def _forced_frame(self, pixel_values: torch.Tensor, state: HCPRCacheState, forced_route: str) -> tuple[list[torch.Tensor], str]:
        if forced_route not in FORCED_ROUTES:
            raise ValueError(f"forced_route must be one of {FORCED_ROUTES}")
        if not state.initialized:
            return self._full_frame(pixel_values, state), "Full"
        self._cache_ready(state)
        f1 = self._run_stage(0, pixel_values)
        if forced_route == "s2_s4":
            features = [f1, state.cached_f2, state.cached_f3, state.cached_f4]
            route = "S2-S4"
        elif forced_route == "s3_s4":
            f2 = self._run_stage(1, f1)
            features = [f1, f2, state.cached_f3, state.cached_f4]
            route = "S3-S4"
        elif forced_route == "s4":
            f2 = self._run_stage(1, f1)
            f3 = self._run_stage(2, f2)
            features = [f1, f2, f3, state.cached_f4]
            route = "S4"
        else:
            features = self._full_frame(pixel_values, state)
            route = "Full"
        self._increment_reuse_ages(state, route) if route != "Full" else None
        return features, route

    def forward(
        self,
        pixel_values: torch.Tensor,
        state: HCPRCacheState | None = None,
        forced_route: str | None = None,
    ) -> tuple[torch.Tensor, HCPRCacheState, dict[str, Any]]:
        """Process one frame and return logits, state, and audit statistics."""
        if state is None:
            state = self.new_state(self.max_stale_run)
        before = dict(self.call_counts)
        drift_values: dict[str, float] = {}
        score_values: dict[str, float] = {}

        if forced_route is not None:
            features, route = self._forced_frame(pixel_values, state, forced_route)
        elif not state.initialized:
            features = self._full_frame(pixel_values, state)
            route = "Full"
        else:
            self._cache_ready(state)
            f1 = self._run_stage(0, pixel_values)
            d1 = self._drift(f1, state.reference_1_to_2)
            q1 = self._score(d1, 2)
            drift_values["d1"] = float(d1.detach().item())
            score_values["q1"] = float(q1.detach().item())
            # A deeper stale cache is only reachable through this prefix.  Its
            # guard therefore propagates upward to preserve the stale bound.
            stage2_guard = any(self._guard_forces(age) for age in (state.age2, state.age3, state.age4))
            if not stage2_guard and bool((q1 <= self.lambda_value).item()):
                features = [f1, state.cached_f2, state.cached_f3, state.cached_f4]
                route = "S2-S4"
                self._increment_reuse_ages(state, route)
            else:
                f2 = self._run_stage(1, f1)
                state.cached_f2 = f2.detach()
                state.reference_1_to_2 = self._pool(f1).detach()
                state.age2 = 0
                d2 = self._drift(f2, state.reference_2_to_3)
                q2 = self._score(d2, 3)
                drift_values["d2"] = float(d2.detach().item())
                score_values["q2"] = float(q2.detach().item())
                stage3_guard = any(self._guard_forces(age) for age in (state.age3, state.age4))
                if not stage3_guard and bool((q2 <= self.lambda_value).item()):
                    features = [f1, f2, state.cached_f3, state.cached_f4]
                    route = "S3-S4"
                    state.age3 += 1
                    state.age4 += 1
                else:
                    f3 = self._run_stage(2, f2)
                    state.cached_f3 = f3.detach()
                    state.reference_2_to_3 = self._pool(f2).detach()
                    state.age3 = 0
                    d3 = self._drift(f3, state.reference_3_to_4)
                    q3 = self._score(d3, 4)
                    drift_values["d3"] = float(d3.detach().item())
                    score_values["q3"] = float(q3.detach().item())
                    if not self._guard_forces(state.age4) and bool((q3 <= self.lambda_value).item()):
                        features = [f1, f2, f3, state.cached_f4]
                        route = "S4"
                        state.age4 += 1
                    else:
                        f4 = self._run_stage(3, f3)
                        state.cached_f4 = f4.detach()
                        state.reference_3_to_4 = self._pool(f3).detach()
                        state.age4 = 0
                        features = [f1, f2, f3, f4]
                        route = "Full"
        self.call_counts["decoder"] += 1
        logits = self.model.decode_head(features)
        deltas = {key: self.call_counts[key] - before[key] for key in self.call_counts}
        return logits, state, {
            "route": route,
            "lambda": self.lambda_value,
            "drift": drift_values,
            "scores": score_values,
            "stage_calls": deltas,
            "cumulative_stage_calls": dict(self.call_counts),
            "age2": state.age2,
            "age3": state.age3,
            "age4": state.age4,
            "persistent_cache_bytes": state.persistent_bytes,
        }
