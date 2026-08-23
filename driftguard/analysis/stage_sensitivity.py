"""Stage-wise stale-feature sensitivity metrics for SegFormer."""

from __future__ import annotations

import torch


def decode_features(model, features: list[torch.Tensor]) -> torch.Tensor:
    """Run the native SegFormer decoder on four NCHW encoder features."""

    if len(features) != 4:
        raise ValueError(f"Expected four stage features, received {len(features)}.")
    return model.decode_head(features)


def _validate_logits(full_logits: torch.Tensor, stale_logits: torch.Tensor) -> None:
    if full_logits.shape != stale_logits.shape:
        raise ValueError(
            "Logit shape mismatch: "
            f"full={tuple(full_logits.shape)}, stale={tuple(stale_logits.shape)}"
        )
    if full_logits.ndim != 4:
        raise ValueError("Logits must be BCHW tensors.")


def prediction_kl_divergence(
    full_logits: torch.Tensor,
    stale_logits: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Mean pixel-wise KL(full prediction || stale prediction)."""

    _validate_logits(full_logits, stale_logits)
    full_log_probs = torch.log_softmax(full_logits.float(), dim=1)
    stale_probs = torch.softmax(stale_logits.float(), dim=1).clamp_min(eps)
    full_probs = full_log_probs.exp()
    kl = (full_probs * (full_log_probs - stale_probs.log())).sum(dim=1)
    return float(kl.mean().item())


def pixel_disagreement_ratio(
    full_logits: torch.Tensor,
    stale_logits: torch.Tensor,
) -> float:
    """Fraction of spatial locations whose argmax class changes."""

    _validate_logits(full_logits, stale_logits)
    full_labels = full_logits.argmax(dim=1)
    stale_labels = stale_logits.argmax(dim=1)
    return float((full_labels != stale_labels).float().mean().item())


def logit_relative_l2(
    full_logits: torch.Tensor,
    stale_logits: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Relative L2 error of stale logits against full logits."""

    _validate_logits(full_logits, stale_logits)
    full_float = full_logits.float()
    stale_float = stale_logits.float()
    return float(
        ((stale_float - full_float).norm(p=2) / (full_float.norm(p=2) + eps)).item()
    )


def stale_feature_metrics(
    model,
    current_features: list[torch.Tensor],
    stale_features: list[torch.Tensor],
    full_logits: torch.Tensor | None = None,
) -> dict[str, float]:
    """Decode one stale-stage variant and compare it with full inference."""

    if full_logits is None:
        full_logits = decode_features(model, current_features)
    stale_logits = decode_features(model, stale_features)
    return {
        "kl_divergence": prediction_kl_divergence(full_logits, stale_logits),
        "pixel_disagreement_ratio": pixel_disagreement_ratio(
            full_logits,
            stale_logits,
        ),
        "logit_relative_l2": logit_relative_l2(full_logits, stale_logits),
    }
