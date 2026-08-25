"""VSPW-124 dataset, metrics, model initialization, and checkpoint helpers."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import Dataset

from driftguard.datasets.vspw import discover_vspw_samples, normalize_vspw_mask

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
NUM_CLASSES = 124
IGNORE_INDEX = 255


def load_vspw_class_names(root: str | Path) -> list[str]:
    mapping = json.loads((Path(root) / "label_num_dic_final.json").read_text(encoding="utf-8"))
    raw_id_to_name = {int(raw_id): name for name, raw_id in mapping.items()}
    if set(raw_id_to_name) != set(range(125)):
        raise ValueError("VSPW taxonomy must contain raw IDs 0..124 exactly")
    return [raw_id_to_name[raw_id] for raw_id in range(1, 125)]


class VSPWFrameDataset(Dataset):
    """Single-frame VSPW training dataset using the official video split."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 512,
        training: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.training = training
        self.samples = discover_vspw_samples(self.root, split=split)
        missing_masks = [sample for sample in self.samples if sample.mask_path is None]
        if missing_masks:
            raise ValueError(f"{split} contains {len(missing_masks)} samples without masks")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as handle:
            image = handle.convert("RGB")
        with Image.open(sample.mask_path) as handle:
            mask = handle.copy()
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        if self.training and torch.rand(()) < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor - IMAGE_MEAN) / IMAGE_STD
        mask_array = normalize_vspw_mask(np.asarray(mask, dtype=np.uint8))
        labels = torch.from_numpy(mask_array.astype(np.int64, copy=False))
        return {
            "pixel_values": image_tensor,
            "labels": labels,
            "video_id": sample.video_id,
            "frame_id": sample.frame_id,
            "image_path": str(sample.image_path),
        }


@dataclass
class ModelInitializationAudit:
    source: str
    loaded_parameter_tensors: int
    missing_keys: list[str]
    unexpected_keys: list[str]
    reinitialized_parameters: list[str]
    trainable_parameters: int
    total_parameters: int


def build_vspw_segformer(
    source: str | Path,
    class_names: list[str],
) -> tuple[torch.nn.Module, ModelInitializationAudit]:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    source = Path(source)
    config = SegformerConfig.from_pretrained(source, local_files_only=True)
    config.num_labels = NUM_CLASSES
    config.id2label = {index: name for index, name in enumerate(class_names)}
    config.label2id = {name: index for index, name in enumerate(class_names)}
    config.semantic_loss_ignore_index = IGNORE_INDEX
    model = SegformerForSemanticSegmentation(config)
    state = load_file(source / "model.safetensors", device="cpu")
    classifier_keys = {
        "decode_head.classifier.weight",
        "decode_head.classifier.bias",
    }
    filtered_state = {key: value for key, value in state.items() if key not in classifier_keys}
    incompatible = model.load_state_dict(filtered_state, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    if set(missing) != classifier_keys or unexpected:
        raise RuntimeError(f"unexpected initialization mismatch: missing={missing}, unexpected={unexpected}")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    audit = ModelInitializationAudit(
        source=str(source),
        loaded_parameter_tensors=len(filtered_state),
        missing_keys=missing,
        unexpected_keys=unexpected,
        reinitialized_parameters=sorted(classifier_keys),
        trainable_parameters=trainable,
        total_parameters=total,
    )
    return model, audit


class StreamingSegmentationMetrics:
    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().to("cpu", torch.int64).reshape(-1)
        target = target.detach().to("cpu", torch.int64).reshape(-1)
        valid = target != self.ignore_index
        prediction = prediction[valid]
        target = target[valid]
        if prediction.numel() and (prediction.min() < 0 or prediction.max() >= self.num_classes):
            raise ValueError("prediction contains an out-of-range class ID")
        encoded = target * self.num_classes + prediction
        self.confusion += torch.bincount(
            encoded, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, Any]:
        matrix = self.confusion.to(torch.float64)
        true_positive = matrix.diag()
        ground_truth = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)
        union = ground_truth + predicted - true_positive
        present = union > 0
        iou = torch.full((self.num_classes,), float("nan"), dtype=torch.float64)
        iou[present] = true_positive[present] / union[present]
        class_accuracy = torch.full_like(iou, float("nan"))
        gt_present = ground_truth > 0
        class_accuracy[gt_present] = true_positive[gt_present] / ground_truth[gt_present]
        total = ground_truth.sum()
        return {
            "miou": float(iou[present].mean()) if present.any() else float("nan"),
            "pixel_accuracy": float(true_positive.sum() / total) if total else float("nan"),
            "mean_class_accuracy": (
                float(class_accuracy[gt_present].mean()) if gt_present.any() else float("nan")
            ),
            "per_class_iou": iou.tolist(),
            "included_classes": present.nonzero().flatten().tolist(),
            "confusion": self.confusion.clone(),
        }


@torch.inference_mode()
def evaluate_model(model, loader, device: torch.device, precision: str = "bf16") -> dict[str, Any]:
    model.eval()
    metrics = StreamingSegmentationMetrics()
    prediction_histogram = torch.zeros(NUM_CLASSES, dtype=torch.int64)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=precision in {"bf16", "fp16"}):
            logits = model(pixel_values=pixel_values).logits
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        prediction = logits.argmax(dim=1)
        metrics.update(prediction, labels)
        prediction_histogram += torch.bincount(
            prediction.detach().cpu().reshape(-1), minlength=NUM_CLASSES
        )
    result = metrics.compute()
    result["prediction_histogram"] = prediction_histogram.tolist()
    return result


def save_training_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    step: int,
    best_miou: float,
    scaler=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "best_miou": best_miou,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(state, path)


def load_training_checkpoint(path, model, optimizer, scheduler, scaler=None) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return {"step": int(state["step"]), "best_miou": float(state["best_miou"])}
