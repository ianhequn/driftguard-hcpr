"""Generate resumable Stage-4 single-block task-damage shards."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.analysis.cheap_observables import (  # noqa: E402
    block_relative_l2,
    entropy_block_observables,
    rgb_block_observables,
)
from driftguard.analysis.sequential_stale import decode_features  # noqa: E402
from driftguard.models.segformer import (  # noqa: E402
    get_stage_features,
    load_segformer,
    prepare_inputs,
)
from driftguard.utils.common import choose_device, set_seed  # noqa: E402
from driftguard.utils.video import iter_video_frames  # noqa: E402


DEFAULT_AGES = (1, 2, 4, 8, 16)


def _patches(current: np.ndarray, reference: np.ndarray, grid: int, size: int) -> np.ndarray:
    residual = np.abs(current.astype(np.float32) / 255.0 - reference.astype(np.float32) / 255.0)
    height, width = residual.shape[:2]
    values = []
    for row in range(grid):
        for col in range(grid):
            crop = residual[
                row * height // grid : (row + 1) * height // grid,
                col * width // grid : (col + 1) * width // grid,
            ]
            tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
            values.append(F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)[0])
    return torch.stack(values).numpy().astype(np.float32)


def _task_metrics(full: torch.Tensor, variants: torch.Tensor, grid: int) -> dict[str, np.ndarray]:
    full = full.float()
    variants = variants.float()
    full_log_probs = torch.log_softmax(full, dim=1)
    full_probs = full_log_probs.exp()
    stale_log_probs = torch.log_softmax(variants, dim=1)
    kl = (full_probs * (full_log_probs - stale_log_probs)).sum(dim=1).mean(dim=(1, 2))
    full_labels = full.argmax(dim=1)
    stale_labels = variants.argmax(dim=1)
    pixel = (full_labels != stale_labels).float()
    global_pixel = pixel.mean(dim=(1, 2))
    full_norm = full.flatten(1).norm(dim=1)
    logit_l2 = (variants - full).flatten(1).norm(dim=1) / full_norm.clamp_min(1e-8)

    output_height, output_width = full.shape[-2:]
    local_kl, local_pixel = [], []
    for index in range(grid * grid):
        row, col = divmod(index, grid)
        r0, r1 = row * output_height // grid, (row + 1) * output_height // grid
        c0, c1 = col * output_width // grid, (col + 1) * output_width // grid
        local_kl.append(kl.new_tensor(0.0) if r1 <= r0 or c1 <= c0 else
                        (full_probs[:, :, r0:r1, c0:c1] *
                         (full_log_probs[:, :, r0:r1, c0:c1] -
                          stale_log_probs[:, :, r0:r1, c0:c1])).sum(dim=1).mean(dim=(1, 2)))
        local_pixel.append(pixel[:, r0:r1, c0:c1].mean(dim=(1, 2)))
    return {
        "global_kl": kl.detach().cpu().numpy(),
        "local_kl": torch.stack(local_kl, dim=1).detach().cpu().numpy(),
        "global_pixel_disagreement": global_pixel.detach().cpu().numpy(),
        "local_pixel_disagreement": torch.stack(local_pixel, dim=1).detach().cpu().numpy(),
        "logit_relative_l2": logit_l2.detach().cpu().numpy(),
    }


def _write_inventory(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="data/real_videos")
    parser.add_argument("--inventory", default="results/R2_E00_video_dataset/video_inventory.csv")
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--output-root", default="results/R2_E01_task_damage_dataset")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--min-frames", type=int, default=32)
    parser.add_argument("--ages", type=int, nargs="+", default=list(DEFAULT_AGES))
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--block-grid", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    if args.block_grid != 4:
        raise ValueError("Round 2 main experiment fixes block-grid=4")
    ages = tuple(sorted({int(value) for value in args.ages}))
    if not ages or any(value <= 0 for value in ages):
        raise ValueError("--ages must contain positive integers")
    inventory = _write_inventory(PROJECT_ROOT / args.inventory)
    if args.max_videos is not None:
        inventory = inventory[: args.max_videos]
    output_root = PROJECT_ROOT / args.output_root
    shard_root = output_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    processor, model = load_segformer(args.model, device=device)
    start = time.perf_counter()
    completed = []

    for record in inventory:
        video_id = record["video_id"]
        shard_path = shard_root / f"{video_id}.npz"
        if shard_path.exists():
            completed.append(video_id)
            continue
        video_path = PROJECT_ROOT / record["path"]
        frames = list(iter_video_frames(video_path, max_frames=args.frames))
        if len(frames) < args.min_frames:
            raise RuntimeError(f"{video_id}: only {len(frames)} frames, need {args.min_frames}")
        frame_rgb_by_id = {frame_id: frame for frame_id, frame in frames}
        features_by_id: dict[int, list[torch.Tensor]] = {}
        logits_by_id: dict[int, torch.Tensor] = {}
        rows: list[dict] = []

        for frame_id, frame_rgb in frames:
            image = Image.fromarray(frame_rgb, mode="RGB")
            inputs = prepare_inputs(processor, image, device)
            with torch.inference_mode():
                outputs = model(
                    pixel_values=inputs["pixel_values"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                current_features = [feature.detach() for feature in get_stage_features(outputs, model=model)]
                full_logits = decode_features(model, current_features).detach()
            features_by_id[frame_id] = current_features
            logits_by_id[frame_id] = full_logits
            for age in ages:
                reference_id = frame_id - age
                if reference_id not in features_by_id:
                    continue
                reference_rgb = frame_rgb_by_id[reference_id]
                cached = features_by_id[reference_id]
                stage = current_features[3]
                stale_stage = cached[3]
                drift = block_relative_l2(stage, stale_stage, grid=args.block_grid)
                cheap = rgb_block_observables(frame_rgb, reference_rgb, grid=args.block_grid)
                entropy = entropy_block_observables(
                    logits_by_id[reference_id],
                    image_height=frame_rgb.shape[0],
                    image_width=frame_rgb.shape[1],
                    grid=args.block_grid,
                )
                patch_array = _patches(frame_rgb, reference_rgb, args.block_grid, args.patch_size)

                variants = []
                for block_index in range(args.block_grid * args.block_grid):
                    variant = stage.clone()
                    row, col = divmod(block_index, args.block_grid)
                    h, w = stage.shape[-2:]
                    r0, r1 = row * h // args.block_grid, (row + 1) * h // args.block_grid
                    c0, c1 = col * w // args.block_grid, (col + 1) * w // args.block_grid
                    variant[:, :, r0:r1, c0:c1] = stale_stage[:, :, r0:r1, c0:c1]
                    variants.append(variant)
                batch_stage = torch.cat(variants, dim=0)
                batch_features = [feature.expand(batch_stage.shape[0], -1, -1, -1) for feature in current_features]
                batch_features[3] = batch_stage
                with torch.inference_mode():
                    stale_logits = decode_features(model, batch_features)
                metrics = _task_metrics(full_logits, stale_logits, args.block_grid)
                for block_index in range(args.block_grid * args.block_grid):
                    rows.append(
                        {
                            "video_id": video_id,
                            "split": record["split"],
                            "frame_id": frame_id,
                            "reference_frame_id": reference_id,
                            "cache_age": age,
                            "block_index": block_index,
                            "block_row": block_index // args.block_grid,
                            "block_col": block_index % args.block_grid,
                            "patch": patch_array[block_index],
                            "metadata": np.asarray(
                                [age, entropy[block_index], cheap[block_index, 2],
                                 cheap[block_index, 0], cheap[block_index, 3]],
                                dtype=np.float32,
                            ),
                            "feature_drift": drift[block_index],
                            "task_damage": metrics["logit_relative_l2"][block_index],
                            "global_kl": metrics["global_kl"][block_index],
                            "local_kl": metrics["local_kl"][block_index, block_index],
                            "global_pixel_disagreement": metrics["global_pixel_disagreement"][block_index],
                            "local_pixel_disagreement": metrics["local_pixel_disagreement"][block_index, block_index],
                        }
                    )
            for old_id in list(features_by_id):
                if old_id < frame_id - max(ages):
                    features_by_id.pop(old_id, None)
                    logits_by_id.pop(old_id, None)
                    frame_rgb_by_id.pop(old_id, None)

        np.savez_compressed(
            shard_path,
            patch=np.stack([row["patch"] for row in rows]).astype(np.float32),
            metadata=np.stack([row["metadata"] for row in rows]).astype(np.float32),
            feature_drift=np.asarray([row["feature_drift"] for row in rows], dtype=np.float32),
            task_damage=np.asarray([row["task_damage"] for row in rows], dtype=np.float32),
            global_kl=np.asarray([row["global_kl"] for row in rows], dtype=np.float32),
            local_kl=np.asarray([row["local_kl"] for row in rows], dtype=np.float32),
            global_pixel_disagreement=np.asarray([row["global_pixel_disagreement"] for row in rows], dtype=np.float32),
            local_pixel_disagreement=np.asarray([row["local_pixel_disagreement"] for row in rows], dtype=np.float32),
            video_id=np.asarray([row["video_id"] for row in rows]),
            split=np.asarray([row["split"] for row in rows]),
            frame_id=np.asarray([row["frame_id"] for row in rows], dtype=np.int32),
            reference_frame_id=np.asarray([row["reference_frame_id"] for row in rows], dtype=np.int32),
            cache_age=np.asarray([row["cache_age"] for row in rows], dtype=np.int32),
            block_index=np.asarray([row["block_index"] for row in rows], dtype=np.int32),
        )
        completed.append(video_id)
        print(f"completed {video_id}: samples={len(rows)}", flush=True)

    train_values = []
    for record in inventory:
        if record["split"] != "train":
            continue
        shard = np.load(shard_root / f"{record['video_id']}.npz")
        train_values.extend(shard["task_damage"].tolist())
    thresholds = {f"p{p}": float(np.quantile(train_values, p / 100.0)) for p in (70, 80, 90)}
    (output_root / "thresholds_train_only.json").write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
    # Add all thresholded labels after thresholds are known. This keeps the
    # threshold source explicit and prevents accidental test-set selection.
    for record in inventory:
        shard_path = shard_root / f"{record['video_id']}.npz"
        shard = dict(np.load(shard_path, allow_pickle=False))
        damage = shard["task_damage"]
        for name, value in thresholds.items():
            shard[f"unsafe_{name}"] = (damage > value).astype(np.uint8)
        np.savez_compressed(shard_path, **shard)
    index_rows = []
    summary_rows = []
    for record in inventory:
        shard = np.load(shard_root / f"{record['video_id']}.npz", allow_pickle=False)
        index_rows.append(
            {
                "video_id": record["video_id"],
                "split": record["split"],
                "shard": str(shard_root.relative_to(PROJECT_ROOT) / f"{record['video_id']}.npz"),
                "sample_count": len(shard["task_damage"]),
            }
        )
        summary_rows.append(
            {
                "video_id": record["video_id"],
                "split": record["split"],
                "sample_count": len(shard["task_damage"]),
                "mean_task_damage": float(shard["task_damage"].mean()),
                "p90_task_damage": float(np.quantile(shard["task_damage"], 0.90)),
            }
        )
    with (output_root / "dataset_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    with (output_root / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (output_root / "config.yaml").write_text(
        f"video_dir: {args.video_dir}\nframes: {args.frames}\nages: {list(ages)}\n"
        f"block_grid: {args.block_grid}\npatch_size: {args.patch_size}\ndevice: {device}\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output_root / "runtime.txt").write_text(f"{time.perf_counter() - start:.6f}\n", encoding="utf-8")
    (output_root / "summary.json").write_text(
        json.dumps({"videos": len(completed), "thresholds_train_only": thresholds}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"videos": len(completed), "thresholds": thresholds}))


if __name__ == "__main__":
    main()
