"""Hugging Face SegFormer loading, inference, and feature normalization."""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
DEFAULT_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/"
    "resolve/main/transformers/tasks/segmentation_input.jpg"
)


def load_segformer(
    model_name: str = DEFAULT_MODEL,
    device: torch.device | str = "cpu",
    local_files_only: bool = False,
):
    """Load a SegFormer processor and semantic segmentation model."""

    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    device = torch.device(device)
    processor = AutoImageProcessor.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        use_fast=False,
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return processor, model


def load_image(
    source: str | Path,
    cache_dir: str | Path = "data/samples",
) -> Image.Image:
    """Load an RGB image from a local path or HTTP(S) URL."""

    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(source_text.split("?", maxsplit=1)[0]).name or "sample.jpg"
        destination = cache_dir / filename
        if not destination.exists():
            urllib.request.urlretrieve(source_text, destination)
        source_path = destination
    else:
        source_path = Path(source_text)

    if not source_path.exists():
        raise FileNotFoundError(f"Image does not exist: {source_path}")
    return Image.open(source_path).convert("RGB")


def prepare_inputs(processor, image: Image.Image, device: torch.device):
    """Convert a PIL image to model inputs on the requested device."""

    inputs = processor(images=image, return_tensors="pt")
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def forward(
    model,
    pixel_values: torch.Tensor,
    output_hidden_states: bool = False,
):
    """Run inference without building an autograd graph."""

    with torch.inference_mode():
        return model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )


def _infer_spatial_shape(token_count: int) -> tuple[int, int]:
    side = int(math.sqrt(token_count))
    if side * side != token_count:
        raise ValueError(
            f"Cannot infer a square feature map from {token_count} tokens. "
            "Use a model output that retains spatial dimensions."
        )
    return side, side


def _normalize_stage_feature(
    feature: torch.Tensor,
    expected_channels: Iterable[int] | None = None,
) -> torch.Tensor:
    """Return one stage feature as [B, C, H, W]."""

    if feature.ndim == 4:
        channels = set(expected_channels or [])
        if channels and feature.shape[1] in channels:
            return feature.contiguous()
        if channels and feature.shape[-1] in channels:
            return feature.permute(0, 3, 1, 2).contiguous()
        if feature.shape[1] <= feature.shape[-1]:
            return feature.contiguous()
        return feature.permute(0, 3, 1, 2).contiguous()

    if feature.ndim != 3:
        raise ValueError(f"Unsupported stage feature rank: {feature.ndim}")

    channels = set(expected_channels or [])
    if channels and feature.shape[-1] in channels:
        batch, tokens, channel_count = feature.shape
        height, width = _infer_spatial_shape(tokens)
        return feature.transpose(1, 2).reshape(batch, channel_count, height, width).contiguous()
    if channels and feature.shape[1] in channels:
        batch, channel_count, tokens = feature.shape
        height, width = _infer_spatial_shape(tokens)
        return feature.reshape(batch, channel_count, height, width).contiguous()

    if feature.shape[-1] <= feature.shape[1]:
        batch, tokens, channel_count = feature.shape
        height, width = _infer_spatial_shape(tokens)
        return feature.transpose(1, 2).reshape(batch, channel_count, height, width).contiguous()

    batch, channel_count, tokens = feature.shape
    height, width = _infer_spatial_shape(tokens)
    return feature.reshape(batch, channel_count, height, width).contiguous()


def get_stage_features(outputs, model=None) -> list[torch.Tensor]:
    """Extract MiT stage 1-4 features as NCHW tensors."""

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None or len(hidden_states) < 4:
        raise RuntimeError(
            "SegFormer output does not contain four hidden states. "
            "Call the model with output_hidden_states=True."
        )

    expected_channels = None
    if model is not None:
        expected_channels = getattr(getattr(model, "config", None), "hidden_sizes", None)
    selected = list(hidden_states[-4:])
    return [
        _normalize_stage_feature(feature, expected_channels=expected_channels)
        for feature in selected
    ]


def get_feature_stats(features: Iterable[torch.Tensor]) -> list[dict[str, Any]]:
    """Collect stable scalar statistics for each stage feature."""

    stats: list[dict[str, Any]] = []
    for index, feature in enumerate(features, start=1):
        detached = feature.detach()
        finite = torch.isfinite(detached)
        stats.append(
            {
                "stage": index,
                "shape": "x".join(str(value) for value in detached.shape),
                "dtype": str(detached.dtype).replace("torch.", ""),
                "mean": float(detached.mean().item()),
                "std": float(detached.std(unbiased=False).item()),
                "min": float(detached.min().item()),
                "max": float(detached.max().item()),
                "has_nan": bool(torch.isnan(detached).any().item()),
                "has_inf": bool(torch.isinf(detached).any().item()),
                "all_finite": bool(finite.all().item()),
            }
        )
    return stats


def predict_segmentation(
    outputs,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upsample logits to the image size and return logits plus class ids."""

    width, height = image_size
    logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    mask = logits.argmax(dim=1)
    return logits, mask


def make_palette(num_labels: int = 256) -> np.ndarray:
    """Create a deterministic RGB palette for class-index visualization."""

    palette = np.zeros((num_labels, 3), dtype=np.uint8)
    for label in range(num_labels):
        value = label
        red = green = blue = 0
        for bit in range(8):
            red |= ((value >> 0) & 1) << (7 - bit)
            green |= ((value >> 1) & 1) << (7 - bit)
            blue |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
        palette[label] = [red, green, blue]
    return palette


def mask_to_images(
    image: Image.Image,
    mask: torch.Tensor,
    num_labels: int,
    alpha: float = 0.45,
) -> tuple[Image.Image, Image.Image]:
    """Convert a class-index mask to a color mask and an overlay."""

    mask_array = mask.detach().cpu().numpy().astype(np.int32)
    palette = make_palette(max(256, num_labels))
    color_mask = palette[mask_array.clip(0, len(palette) - 1)]
    mask_image = Image.fromarray(color_mask, mode="RGB")
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = (base * (1.0 - alpha) + color_mask.astype(np.float32) * alpha).clip(0, 255)
    return mask_image, Image.fromarray(overlay.astype(np.uint8), mode="RGB")
