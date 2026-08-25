"""Smoke, calibrate, and train the Round 7.6 VSPW SegFormer-B0 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.training.vspw_segformer import (  # noqa: E402
    IGNORE_INDEX,
    NUM_CLASSES,
    StreamingSegmentationMetrics,
    VSPWFrameDataset,
    build_vspw_segformer,
    load_training_checkpoint,
    load_vspw_class_names,
    save_training_checkpoint,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loader_kwargs(workers: int) -> dict:
    kwargs = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return kwargs


def make_scheduler(optimizer, schedule_steps: int, warmup_ratio: float):
    warmup_steps = max(1, round(schedule_steps * warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, schedule_steps - warmup_steps))
        return (1.0 - progress) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor), warmup_steps


def precision_context(precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def validate(model, loader, device, precision: str, max_batches: int | None = None):
    model.eval()
    metrics = StreamingSegmentationMetrics()
    histogram = torch.zeros(NUM_CLASSES, dtype=torch.int64)
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with precision_context(precision):
            logits = model(pixel_values=images).logits
        logits = torch.nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        prediction = logits.argmax(dim=1)
        metrics.update(prediction, labels)
        histogram += torch.bincount(prediction.cpu().reshape(-1), minlength=NUM_CLASSES)
    result = metrics.compute()
    result.pop("confusion")
    result["prediction_histogram"] = histogram.tolist()
    result["elapsed_seconds"] = time.perf_counter() - started
    result["max_batches"] = max_batches
    model.train()
    return result


def append_history(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_step_checkpoint(root, step, model, optimizer, scheduler, best_miou, scaler):
    path = root / f"step_{step:06d}" / "training_state.pt"
    save_training_checkpoint(path, model, optimizer, scheduler, step, best_miou, scaler)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "calibrate", "train"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--schedule-steps", type=int, default=40000)
    parser.add_argument("--validation-every", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Round 7.6 training requires CUDA")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = torch.device("cuda")
    class_names = load_vspw_class_names(args.dataset_root)
    model, initialization = build_vspw_segformer(args.source, class_names)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler, warmup_steps = make_scheduler(optimizer, args.schedule_steps, args.warmup_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=args.precision == "fp16")

    start_step = 0
    best_miou = -math.inf
    if args.resume:
        resumed = load_training_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        start_step = resumed["step"]
        best_miou = resumed["best_miou"]

    train_dataset = VSPWFrameDataset(args.dataset_root, "train", training=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs(args.workers),
    )
    validation_loader = None
    if args.mode in {"smoke", "train"}:
        validation_dataset = VSPWFrameDataset(args.dataset_root, "val", training=False)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_kwargs(args.workers),
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "arguments": vars(args) | {
            "dataset_root": str(args.dataset_root),
            "source": str(args.source),
            "output_root": str(args.output_root),
            "resume": str(args.resume) if args.resume else None,
        },
        "initialization": initialization.__dict__,
        "warmup_steps": warmup_steps,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_loader.dataset) if validation_loader else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic": torch.backends.cudnn.deterministic,
    }
    (args.output_root / "run_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    history_path = args.output_root / "training_history.csv"
    checkpoint_root = PROJECT_ROOT / "checkpoints/vspw_segformer_b0"
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    measured_step_times: list[float] = []
    interval_loss = 0.0
    interval_count = 0

    for step in range(start_step + 1, args.steps + 1):
        step_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        valid_labels = labels[labels != IGNORE_INDEX]
        if valid_labels.numel() == 0 or valid_labels.min() < 0 or valid_labels.max() >= NUM_CLASSES:
            raise RuntimeError("training labels violate the VSPW-124 contract")
        with precision_context(args.precision):
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss / args.accumulation
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        scaler.scale(loss).backward()
        if step % args.accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_started
        if step > start_step + 20:
            measured_step_times.append(elapsed)
        interval_loss += float(loss.item()) * args.accumulation
        interval_count += 1

        should_log = step % 10 == 0 or step == args.steps
        if should_log:
            mean_loss = interval_loss / interval_count
            row = {
                "step": step,
                "epoch_equivalent": step * args.batch_size / len(train_dataset),
                "train_loss": mean_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "images_seen": step * args.batch_size,
                "wall_time": time.perf_counter() - run_started,
                "validation_mIoU": "",
                "pixel_accuracy": "",
                "peak_vram": torch.cuda.max_memory_allocated() / (1024**3),
            }
            append_history(history_path, row)
            print(json.dumps(row), flush=True)
            interval_loss = 0.0
            interval_count = 0

        if args.mode == "train" and step % args.validation_every == 0:
            validation = validate(model, validation_loader, device, args.precision)
            validation["step"] = step
            (args.output_root / f"validation_{step:06d}.json").write_text(
                json.dumps(validation, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
            )
            append_history(
                history_path,
                {
                    "step": step,
                    "epoch_equivalent": step * args.batch_size / len(train_dataset),
                    "train_loss": "",
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "images_seen": step * args.batch_size,
                    "wall_time": time.perf_counter() - run_started,
                    "validation_mIoU": validation["miou"],
                    "pixel_accuracy": validation["pixel_accuracy"],
                    "peak_vram": torch.cuda.max_memory_allocated() / (1024**3),
                },
            )
            improved = validation["miou"] > best_miou
            if improved:
                best_miou = validation["miou"]
            checkpoint = save_step_checkpoint(
                checkpoint_root, step, model, optimizer, scheduler, best_miou, scaler
            )
            if improved:
                best_dir = checkpoint_root / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(checkpoint, best_dir / "training_state.pt")
                (best_dir / "selection.json").write_text(
                    json.dumps({"step": step, "miou": best_miou}, indent=2) + "\n",
                    encoding="utf-8",
                )

        if args.mode == "train" and step % args.save_every == 0 and step % args.validation_every:
            save_step_checkpoint(checkpoint_root, step, model, optimizer, scheduler, best_miou, scaler)

    smoke_validation = None
    if args.mode == "smoke":
        smoke_validation = validate(
            model, validation_loader, device, args.precision, max_batches=1
        )
        save_training_checkpoint(
            args.output_root / "smoke_resume.pt",
            model,
            optimizer,
            scheduler,
            args.steps,
            best_miou,
            scaler,
        )

    summary = {
        "mode": args.mode,
        "start_step": start_step,
        "end_step": args.steps,
        "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size * args.accumulation,
        "precision": args.precision,
        "mean_step_time_seconds": float(np.mean(measured_step_times)),
        "images_per_second": (
            args.batch_size / float(np.mean(measured_step_times)) if measured_step_times else None
        ),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "elapsed_seconds": time.perf_counter() - run_started,
        "smoke_validation": smoke_validation,
        "best_miou": None if best_miou == -math.inf else best_miou,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
