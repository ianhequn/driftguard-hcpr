"""Fixed-budget Stage-4 refresh simulation for Round 2."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.analysis.cheap_observables import block_relative_l2, entropy_block_observables, rgb_block_observables  # noqa: E402
from driftguard.analysis.sequential_stale import decode_features  # noqa: E402
from driftguard.models.risk_predictor import TinyRiskCNN  # noqa: E402
from driftguard.models.segformer import get_stage_features, load_segformer, prepare_inputs  # noqa: E402
from driftguard.utils.common import choose_device, set_seed  # noqa: E402
from driftguard.utils.video import iter_video_frames  # noqa: E402
from generate_round2_task_dataset import _patches, _task_metrics  # noqa: E402


RATIOS = (0.125, 0.25, 0.375, 0.5, 0.75, 1.0)
POLICIES = ("random", "residual_mean", "residual_p90", "age", "tiny_risk_cnn", "oracle")


def _select(scores: np.ndarray, ratio: float) -> np.ndarray:
    count = int(np.ceil(len(scores) * ratio))
    mask = np.zeros(len(scores), dtype=bool)
    mask[np.argsort(scores)[-count:]] = True
    return mask


def _replace(cache: torch.Tensor, current: torch.Tensor, mask: np.ndarray, grid: int) -> torch.Tensor:
    updated = cache.clone()
    h, w = current.shape[-2:]
    for index, selected in enumerate(mask):
        if not selected:
            continue
        row, col = divmod(index, grid)
        r0, r1 = row * h // grid, (row + 1) * h // grid
        c0, c1 = col * w // grid, (col + 1) * w // grid
        updated[:, :, r0:r1, c0:c1] = current[:, :, r0:r1, c0:c1]
    return updated


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="data/real_videos")
    parser.add_argument("--inventory", default="results/R2_E00_video_dataset/video_inventory.csv")
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--predictor", default="results/R2_E02_predictor/best_model.pt")
    parser.add_argument("--output-root", default="results/R2_E04_refresh_simulation")
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--random-seeds", type=int, default=20)
    parser.add_argument("--decoder-batch", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(PROJECT_ROOT / args.predictor, map_location="cpu")
    predictor = TinyRiskCNN(metadata_dim=len(checkpoint["metadata_columns"]))
    predictor.load_state_dict(checkpoint["state_dict"])
    predictor.to(device).eval()
    metadata_mean = np.asarray(checkpoint["metadata_mean"], dtype=np.float32)
    metadata_std = np.asarray(checkpoint["metadata_std"], dtype=np.float32)
    metadata_std[metadata_std < 1e-8] = 1.0

    with (PROJECT_ROOT / args.inventory).open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    processor, model = load_segformer(args.model, device=device)
    rows = []
    start = time.perf_counter()

    records = [record for record in records if record["split"] == "test"]
    for record in records:
        video_path = PROJECT_ROOT / record["path"]
        frames = list(iter_video_frames(video_path, max_frames=args.max_frames))
        previous_frame = None
        previous_logits = None
        states = {}
        for seed in range(args.random_seeds):
            for ratio in RATIOS:
                states[("random", seed, ratio)] = {"cache": None, "age": np.zeros(16, dtype=np.float32), "rng": random.Random(args.seed + seed)}
        for policy in ("residual_mean", "residual_p90", "age", "tiny_risk_cnn", "oracle"):
            for ratio in RATIOS:
                states[(policy, 0, ratio)] = {"cache": None, "age": np.zeros(16, dtype=np.float32), "rng": None}

        for frame_id, frame_rgb in frames:
            image = Image.fromarray(frame_rgb, mode="RGB")
            inputs = prepare_inputs(processor, image, device)
            with torch.inference_mode():
                outputs = model(pixel_values=inputs["pixel_values"], output_hidden_states=True, return_dict=True)
                current_features = [feature.detach() for feature in get_stage_features(outputs, model=model)]
                full_logits = decode_features(model, current_features).detach()
            current_stage = current_features[3]
            if previous_frame is None:
                previous_frame = frame_rgb
                previous_logits = full_logits
                for state in states.values():
                    state["cache"] = current_stage.clone()
                continue

            cheap = rgb_block_observables(frame_rgb, previous_frame, grid=4)
            entropy = entropy_block_observables(previous_logits, frame_rgb.shape[0], frame_rgb.shape[1], grid=4)
            patches = _patches(frame_rgb, previous_frame, 4, 16)
            metadata = np.column_stack(
                [np.zeros(16, dtype=np.float32), entropy, np.full(16, cheap[:, 2][0], dtype=np.float32), cheap[:, 0], cheap[:, 3]]
            ).astype(np.float32)
            metadata[:, 0] = 0.0
            variants, descriptors = [], []

            for key, state in states.items():
                policy, seed, ratio = key
                if state["cache"] is None:
                    state["cache"] = current_stage.clone()
                true_drift = block_relative_l2(current_stage, state["cache"], grid=4)
                metadata[:, 0] = state["age"]
                if policy == "random":
                    scores = np.asarray([state["rng"].random() for _ in range(16)], dtype=np.float32)
                elif policy == "residual_mean":
                    scores = cheap[:, 0]
                elif policy == "residual_p90":
                    scores = np.quantile(patches.reshape(16, -1), 0.90, axis=1)
                elif policy == "age":
                    scores = state["age"].copy()
                elif policy == "tiny_risk_cnn":
                    with torch.inference_mode():
                        scores = torch.sigmoid(
                            predictor(
                                torch.from_numpy(patches).to(device),
                                torch.from_numpy((metadata - metadata_mean) / metadata_std).to(device),
                            )
                        ).cpu().numpy()
                elif policy == "oracle":
                    scores = true_drift
                else:
                    raise ValueError(policy)
                mask = _select(scores, ratio)
                state["cache"] = _replace(state["cache"], current_stage, mask, 4)
                state["age"] += 1.0
                state["age"][mask] = 0.0
                variants.append(state["cache"])
                descriptors.append((policy, seed, ratio, int(mask.sum()), true_drift))

            # Keep the policy state space large enough for 20 random seeds,
            # but decode in small micro-batches to bound GPU memory.
            for start_index in range(0, len(variants), args.decoder_batch):
                end_index = start_index + args.decoder_batch
                batch_stage = torch.cat(variants[start_index:end_index], dim=0)
                batch_features = [
                    feature.expand(batch_stage.shape[0], -1, -1, -1)
                    for feature in current_features
                ]
                batch_features[3] = batch_stage
                with torch.inference_mode():
                    stale_logits = decode_features(model, batch_features)
                metrics = _task_metrics(full_logits, stale_logits, 4)
                for index, (policy, seed, ratio, refresh_blocks, _) in enumerate(
                    descriptors[start_index:end_index]
                ):
                    rows.append(
                        {
                            "video_id": record["video_id"],
                            "split": record["split"],
                            "frame_id": frame_id,
                            "policy": policy,
                            "seed": seed,
                            "refresh_ratio": ratio,
                            "refresh_blocks": refresh_blocks,
                            "global_kl": float(metrics["global_kl"][index]),
                            "logit_relative_l2": float(metrics["logit_relative_l2"][index]),
                            "global_pixel_disagreement": float(
                                metrics["global_pixel_disagreement"][index]
                            ),
                        }
                    )
                del batch_stage, batch_features, stale_logits
            del variants, descriptors
            previous_frame = frame_rgb
            previous_logits = full_logits
        print(f"completed {record['video_id']}", flush=True)

    output_root = PROJECT_ROOT / args.output_root
    _write_csv(output_root / "policy_metrics.csv", rows)
    summary = []
    for policy in POLICIES:
        for ratio in RATIOS:
            group = [row for row in rows if row["policy"] == policy and float(row["refresh_ratio"]) == ratio and row["split"] == "test"]
            if not group:
                continue
            summary.append(
                {
                    "policy": policy,
                    "refresh_ratio": ratio,
                    "samples": len(group),
                    "video_count": len({row["video_id"] for row in group}),
                    "global_kl_mean": float(np.mean([row["global_kl"] for row in group])),
                    "global_kl_std": float(np.std([row["global_kl"] for row in group])),
                    "logit_relative_l2_mean": float(np.mean([row["logit_relative_l2"] for row in group])),
                    "logit_relative_l2_std": float(np.std([row["logit_relative_l2"] for row in group])),
                    "global_pixel_disagreement_mean": float(np.mean([row["global_pixel_disagreement"] for row in group])),
                    "global_pixel_disagreement_std": float(np.std([row["global_pixel_disagreement"] for row in group])),
                }
            )
    _write_csv(output_root / "summary_test.csv", summary)
    for metric, ylabel, filename in (
        ("logit_relative_l2_mean", "Logit relative L2 damage", "refresh_ratio_vs_logit_damage.png"),
        ("global_pixel_disagreement_mean", "Pixel disagreement", "refresh_ratio_vs_pixel_disagreement.png"),
        ("global_kl_mean", "KL divergence", "refresh_ratio_vs_kl.png"),
    ):
        plt.figure(figsize=(8, 5))
        for policy in POLICIES:
            values = sorted([row for row in summary if row["policy"] == policy], key=lambda row: row["refresh_ratio"])
            if values:
                plt.plot([row["refresh_ratio"] for row in values], [row[metric] for row in values], marker="o", label=policy)
        plt.xlabel("Refresh ratio")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_root / filename, dpi=160)
        plt.close()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config.yaml").write_text(
        f"max_frames: {args.max_frames}\nrandom_seeds: {args.random_seeds}\n"
        f"decoder_batch: {args.decoder_batch}\ndevice: {device}\nratios: {list(RATIOS)}\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output_root / "runtime.txt").write_text(f"{time.perf_counter() - start:.6f}\n", encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps({"rows": len(rows), "test_summary": summary}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "test_summary": summary}))


if __name__ == "__main__":
    main()
