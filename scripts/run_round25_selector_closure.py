"""Unified Round 2.5 selector comparison under corrected refresh semantics."""

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

from driftguard.analysis.cheap_observables import (  # noqa: E402
    entropy_block_observables,
    rgb_block_observables,
)
from driftguard.analysis.refresh_semantics import (  # noqa: E402
    RefreshState,
    select_top_k,
    validate_analysis_only_policies,
)
from driftguard.analysis.sequential_stale import decode_features  # noqa: E402
from driftguard.models.risk_predictor import TinyRiskCNN  # noqa: E402
from driftguard.models.segformer import (  # noqa: E402
    get_stage_features,
    load_segformer,
    prepare_inputs,
)
from driftguard.utils.common import choose_device, set_seed  # noqa: E402
from driftguard.utils.video import iter_video_frames  # noqa: E402
from generate_round2_task_dataset import _patches, _task_metrics  # noqa: E402
from train_round25_old_mlp import HistoricalTinyMLP  # noqa: E402


GRID = 4
BLOCKS = GRID * GRID
RATIOS = (0.125, 0.25, 0.375, 0.5, 0.75, 1.0)
DEPLOYABLE_POLICIES = (
    "random",
    "residual_mean",
    "residual_p90",
    "age",
    "old_tiny_mlp",
    "cnn_all",
    "cnn_patch_age",
)
ALL_POLICIES = DEPLOYABLE_POLICIES + ("oracle",)
validate_analysis_only_policies(DEPLOYABLE_POLICIES)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _replace_state(state: RefreshState, current_stage: torch.Tensor, mask: np.ndarray) -> None:
    state.apply_refresh(current_stage, mask, grid=GRID)


def _oracle_task_scores(
    model,
    current_features: list[torch.Tensor],
    cache_stage: torch.Tensor,
    full_logits: torch.Tensor,
    decoder_batch: int,
) -> np.ndarray:
    """Score stale single-block interventions by actual decoder task damage."""

    current_stage = current_features[3]
    variants = []
    height, width = current_stage.shape[-2:]
    for block_index in range(BLOCKS):
        variant = current_stage.clone()
        row, col = divmod(block_index, GRID)
        r0, r1 = row * height // GRID, (row + 1) * height // GRID
        c0, c1 = col * width // GRID, (col + 1) * width // GRID
        variant[:, :, r0:r1, c0:c1] = cache_stage[:, :, r0:r1, c0:c1]
        variants.append(variant)

    scores = []
    for start in range(0, BLOCKS, decoder_batch):
        batch_stage = torch.cat(variants[start : start + decoder_batch], dim=0)
        batch_features = [
            feature.expand(batch_stage.shape[0], -1, -1, -1)
            for feature in current_features
        ]
        batch_features[3] = batch_stage
        with torch.inference_mode():
            stale_logits = decode_features(model, batch_features)
        metrics = _task_metrics(full_logits, stale_logits, GRID)
        scores.extend(metrics["logit_relative_l2"].tolist())
        del batch_stage, batch_features, stale_logits
    return np.asarray(scores, dtype=np.float32)


def _load_old_mlp(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    model = HistoricalTinyMLP(len(checkpoint["feature_columns"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    std[std < 1e-8] = 1.0
    return model, mean, std


def _load_cnn(path: Path, metadata_dim: int, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    model = TinyRiskCNN(metadata_dim=metadata_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    mean = np.asarray(checkpoint["metadata_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["metadata_std"], dtype=np.float32)
    std[std < 1e-8] = 1.0
    indices = checkpoint.get("metadata_indices", list(range(metadata_dim)))
    return model, mean, std, list(indices)


def _predict_old(
    model,
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    old_values = values[:, [3, 0, 1, 2, 4]]
    with torch.inference_mode():
        return model(
            torch.from_numpy((old_values - mean) / std).to(device)
        ).detach().cpu().numpy()


def _predict_cnn(
    model,
    patches: np.ndarray,
    metadata: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    indices: list[int],
    device: torch.device,
) -> np.ndarray:
    selected = metadata[:, indices]
    with torch.inference_mode():
        return model(
            torch.from_numpy(patches).to(device),
            torch.from_numpy((selected - mean) / std).to(device),
        ).detach().cpu().numpy()


def _ranking_overlap(scores: np.ndarray, oracle: np.ndarray, count: int) -> float:
    selected = set(np.flatnonzero(select_top_k(scores, count / BLOCKS)).tolist())
    target = set(np.flatnonzero(select_top_k(oracle, count / BLOCKS)).tolist())
    return float(len(selected & target) / count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="data/real_videos")
    parser.add_argument("--inventory", default="results/R2_E00_video_dataset/video_inventory.csv")
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument(
        "--old-mlp",
        default="results/R25_E01_selector_closure/old_tiny_mlp_taskrisk.pt",
    )
    parser.add_argument("--cnn-all", default="results/R2_E02_predictor/best_model.pt")
    parser.add_argument("--cnn-patch-age", default="results/R2_E03_ablation/patch_age.pt")
    parser.add_argument(
        "--output-root",
        default="results/R25_E01_selector_closure",
    )
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--random-seeds", type=int, default=50)
    parser.add_argument("--decoder-batch", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.random_seeds < 50:
        raise ValueError("Round 2.5 requires at least 50 random seeds")
    set_seed(args.seed)
    device = choose_device(args.device)
    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    old_mlp, old_mean, old_std = _load_old_mlp(PROJECT_ROOT / args.old_mlp, device)
    cnn_all, cnn_all_mean, cnn_all_std, cnn_all_indices = _load_cnn(
        PROJECT_ROOT / args.cnn_all,
        metadata_dim=5,
        device=device,
    )
    cnn_patch_age, cnn_pa_mean, cnn_pa_std, cnn_pa_indices = _load_cnn(
        PROJECT_ROOT / args.cnn_patch_age,
        metadata_dim=1,
        device=device,
    )

    with (PROJECT_ROOT / args.inventory).open(encoding="utf-8", newline="") as handle:
        records = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    processor, model = load_segformer(args.model, device=device)
    rows: list[dict] = []
    ranking_rows: list[dict] = []
    start_time = time.perf_counter()

    for record in records:
        video_path = PROJECT_ROOT / record["path"]
        frames = list(iter_video_frames(video_path, max_frames=args.max_frames))
        states: dict[tuple[str, int, float], RefreshState] = {}
        random_generators: dict[tuple[str, int, float], random.Random] = {}
        for seed in range(args.random_seeds):
            for ratio in RATIOS:
                states[("random", seed, ratio)] = None
                random_generators[("random", seed, ratio)] = random.Random(
                    args.seed + seed * 1009 + int(ratio * 1000)
                )
        for policy in DEPLOYABLE_POLICIES[1:]:
            for ratio in RATIOS:
                states[(policy, 0, ratio)] = None
        for ratio in RATIOS:
            states[("oracle", 0, ratio)] = None
        previous_frame = None
        previous_logits = None

        for frame_id, frame_rgb in frames:
            image = Image.fromarray(frame_rgb, mode="RGB")
            inputs = prepare_inputs(processor, image, device)
            with torch.inference_mode():
                outputs = model(
                    pixel_values=inputs["pixel_values"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                current_features = [
                    feature.detach()
                    for feature in get_stage_features(outputs, model=model)
                ]
                full_logits = decode_features(model, current_features).detach()
            current_stage = current_features[3]

            if previous_frame is None:
                for key in states:
                    states[key] = RefreshState.initialize(current_stage, BLOCKS)
                previous_frame = frame_rgb
                previous_logits = full_logits
                continue

            cheap = rgb_block_observables(frame_rgb, previous_frame, grid=GRID)
            entropy = entropy_block_observables(
                previous_logits,
                image_height=frame_rgb.shape[0],
                image_width=frame_rgb.shape[1],
                grid=GRID,
            )
            patches = _patches(frame_rgb, previous_frame, GRID, 16)
            base_metadata = np.column_stack(
                [
                    np.zeros(BLOCKS, dtype=np.float32),
                    entropy,
                    np.full(BLOCKS, cheap[0, 2], dtype=np.float32),
                    cheap[:, 0],
                    cheap[:, 3],
                ]
            ).astype(np.float32)

            oracle_scores: dict[float, np.ndarray] = {}
            for ratio in RATIOS:
                oracle_state = states[("oracle", 0, ratio)]
                oracle_scores[ratio] = _oracle_task_scores(
                    model,
                    current_features,
                    oracle_state.cache,
                    full_logits,
                    args.decoder_batch,
                )

            variants = []
            descriptors = []
            for key, state in states.items():
                policy, seed, ratio = key
                metadata = base_metadata.copy()
                metadata[:, 0] = state.age
                if policy == "random":
                    scores = np.asarray(
                        [
                            random_generators[key].random()
                            for _ in range(BLOCKS)
                        ],
                        dtype=np.float32,
                    )
                elif policy == "residual_mean":
                    scores = cheap[:, 0]
                elif policy == "residual_p90":
                    scores = np.quantile(patches.reshape(BLOCKS, -1), 0.90, axis=1)
                elif policy == "age":
                    scores = state.age.copy()
                elif policy == "old_tiny_mlp":
                    scores = _predict_old(old_mlp, metadata, old_mean, old_std, device)
                elif policy == "cnn_all":
                    scores = _predict_cnn(
                        cnn_all,
                        patches,
                        metadata,
                        cnn_all_mean,
                        cnn_all_std,
                        cnn_all_indices,
                        device,
                    )
                elif policy == "cnn_patch_age":
                    scores = _predict_cnn(
                        cnn_patch_age,
                        patches,
                        metadata,
                        cnn_pa_mean,
                        cnn_pa_std,
                        cnn_pa_indices,
                        device,
                    )
                elif policy == "oracle":
                    scores = oracle_scores[ratio]
                else:
                    raise ValueError(policy)

                mask = select_top_k(scores, ratio)
                _replace_state(state, current_stage, mask)
                variants.append(state.cache)
                descriptors.append(
                    (
                        policy,
                        seed,
                        ratio,
                        int(mask.sum()),
                        scores,
                        oracle_scores[ratio],
                    )
                )
                if ratio in (0.25, 0.375, 0.5):
                    ranking_rows.append(
                        {
                            "video_id": record["video_id"],
                            "motion_level": record["motion_level"],
                            "frame_id": frame_id,
                            "policy": policy,
                            "seed": seed,
                            "refresh_ratio": ratio,
                            "top4_overlap": _ranking_overlap(scores, oracle_scores[ratio], 4),
                            "top6_overlap": _ranking_overlap(scores, oracle_scores[ratio], 6),
                            "top8_overlap": _ranking_overlap(scores, oracle_scores[ratio], 8),
                        }
                    )

            for start in range(0, len(variants), args.decoder_batch):
                end = start + args.decoder_batch
                batch_stage = torch.cat(variants[start:end], dim=0)
                batch_features = [
                    feature.expand(batch_stage.shape[0], -1, -1, -1)
                    for feature in current_features
                ]
                batch_features[3] = batch_stage
                with torch.inference_mode():
                    stale_logits = decode_features(model, batch_features)
                metrics = _task_metrics(full_logits, stale_logits, GRID)
                for index, (policy, seed, ratio, refresh_count, _, _) in enumerate(
                    descriptors[start:end]
                ):
                    rows.append(
                        {
                            "video_id": record["video_id"],
                            "motion_level": record["motion_level"],
                            "frame_id": frame_id,
                            "policy": policy,
                            "seed": seed,
                            "refresh_ratio": ratio,
                            "refresh_blocks": refresh_count,
                            "global_kl": float(metrics["global_kl"][index]),
                            "logit_relative_l2": float(
                                metrics["logit_relative_l2"][index]
                            ),
                            "global_pixel_disagreement": float(
                                metrics["global_pixel_disagreement"][index]
                            ),
                        }
                    )
                del batch_stage, batch_features, stale_logits

            previous_frame = frame_rgb
            previous_logits = full_logits
        print(f"completed {record['video_id']}", flush=True)

    _write_csv(output_root / "policy_metrics.csv", rows)
    _write_csv(output_root / "ranking_metrics.csv", ranking_rows)
    summary = []
    for policy in ALL_POLICIES:
        for ratio in RATIOS:
            group = [
                row
                for row in rows
                if row["policy"] == policy and float(row["refresh_ratio"]) == ratio
            ]
            if not group:
                continue
            summary.append(
                {
                    "policy": policy,
                    "refresh_ratio": ratio,
                    "samples": len(group),
                    "video_count": len({row["video_id"] for row in group}),
                    "global_kl_mean": float(
                        np.mean([float(row["global_kl"]) for row in group])
                    ),
                    "global_kl_std": float(
                        np.std([float(row["global_kl"]) for row in group])
                    ),
                    "logit_relative_l2_mean": float(
                        np.mean(
                            [float(row["logit_relative_l2"]) for row in group]
                        )
                    ),
                    "logit_relative_l2_std": float(
                        np.std(
                            [float(row["logit_relative_l2"]) for row in group]
                        )
                    ),
                    "global_pixel_disagreement_mean": float(
                        np.mean(
                            [
                                float(row["global_pixel_disagreement"])
                                for row in group
                            ]
                        )
                    ),
                }
            )
    _write_csv(output_root / "summary.csv", summary)
    for metric, ylabel, filename in (
        (
            "logit_relative_l2_mean",
            "Mean test logit relative L2",
            "refresh_budget_vs_task_damage.png",
        ),
    ):
        plt.figure(figsize=(8, 5))
        for policy in ALL_POLICIES:
            values = sorted(
                [row for row in summary if row["policy"] == policy],
                key=lambda row: row["refresh_ratio"],
            )
            if values:
                plt.plot(
                    [row["refresh_ratio"] for row in values],
                    [row[metric] for row in values],
                    marker="o",
                    label=policy,
                )
        plt.xlabel("Refresh ratio")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_root / filename, dpi=160)
        plt.close()

    (output_root / "config.yaml").write_text(
        f"max_frames: {args.max_frames}\nrandom_seeds: {args.random_seeds}\n"
        f"decoder_batch: {args.decoder_batch}\nratios: {list(RATIOS)}\n"
        f"oracle_score: single_block_task_damage_logit_relative_l2\n"
        f"device: {device}\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    (output_root / "runtime.txt").write_text(
        f"{time.perf_counter() - start_time:.6f}\n",
        encoding="utf-8",
    )
    (output_root / "summary.json").write_text(
        json.dumps({"rows": len(rows), "summary": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "summary": summary}))


if __name__ == "__main__":
    main()
