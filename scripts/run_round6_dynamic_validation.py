"""Select whole-Stage-4 visual refresh policies on validation only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.analysis.cheap_observables import (  # noqa: E402
    rgb_block_observables,
)
from driftguard.models.risk_predictor import TinyRiskCNN  # noqa: E402
from driftguard.models.segformer import (  # noqa: E402
    get_stage_features,
    load_segformer,
    prepare_inputs,
)
from driftguard.profiling.selector_gpu import (  # noqa: E402
    gpu_rgb_residual,
    resize_candidate_blocks,
    vectorized_all_blocks,
)
from driftguard.structured_reuse.pipeline import Stage4CacheState  # noqa: E402
from driftguard.structured_reuse.policies import (  # noqa: E402
    StageAgeVisualPolicy,
    VisualRiskPolicy,
    aggregate_frame_risk,
)
from driftguard.utils.common import set_seed  # noqa: E402
from driftguard.utils.video import iter_video_frames  # noqa: E402
from generate_round2_task_dataset import _task_metrics  # noqa: E402


AGGREGATIONS = ("max", "mean", "p90", "topk_mean")
TARGET_REFRESH_RATES = (0.25, 0.40, 0.50, 0.60, 0.75)
AMAX_VALUES = (1, 2, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        default="results/R2_E00_video_dataset/video_inventory.csv",
    )
    parser.add_argument(
        "--predictor",
        default="results/R2_E03_ablation/patch_age.pt",
    )
    parser.add_argument(
        "--model",
        default="/root/autodl-tmp/models/segformer-b0",
    )
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        default="results/R6_E03_visual_stage",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_predictor(
    path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    checkpoint = torch.load(path, map_location="cpu")
    model = TinyRiskCNN(metadata_dim=1)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    mean = np.asarray(checkpoint["metadata_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["metadata_std"], dtype=np.float32)
    std[std < 1e-8] = 1.0
    return model, mean, std


def frame_risk(
    predictor,
    predictor_mean: np.ndarray,
    predictor_std: np.ndarray,
    current: torch.Tensor,
    previous: torch.Tensor,
    stage_age: int,
) -> tuple[np.ndarray, dict[str, float]]:
    residual = gpu_rgb_residual(current, previous)
    patches = resize_candidate_blocks(
        vectorized_all_blocks(residual, grid=4),
        size=16,
    )
    metadata_np = np.full((16, 1), float(stage_age), dtype=np.float32)
    metadata = torch.from_numpy(
        (metadata_np - predictor_mean) / predictor_std
    ).to(current.device)
    with torch.inference_mode():
        block_scores = (
            predictor(patches, metadata)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
    return block_scores, {
        "rgb_residual_mean": float(residual.mean().item()),
        "rgb_residual_p90": float(
            torch.quantile(residual.float(), 0.90).item()
        ),
    }


def block_scores_by_age(
    predictor,
    predictor_mean: np.ndarray,
    predictor_std: np.ndarray,
    current: torch.Tensor,
    previous: torch.Tensor,
    max_age: int,
) -> dict[str, list[float]]:
    """Score the same current/previous RGB pair for every deployable age."""

    residual = gpu_rgb_residual(current, previous)
    patches = resize_candidate_blocks(
        vectorized_all_blocks(residual, grid=4),
        size=16,
    )
    ages = np.arange(max_age + 1, dtype=np.float32)
    repeated_patches = patches.repeat(max_age + 1, 1, 1, 1)
    metadata_np = np.repeat(ages[:, None], 16, axis=0).reshape(-1, 1)
    metadata = torch.from_numpy(
        (metadata_np - predictor_mean) / predictor_std
    ).to(current.device)
    with torch.inference_mode():
        values = (
            predictor(repeated_patches, metadata)
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(max_age + 1, 16)
        )
    return {
        str(age): values[age].astype(np.float32).tolist()
        for age in range(max_age + 1)
    }


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty_like(order, dtype=np.float64)
        result[order] = np.arange(len(values), dtype=np.float64)
        return result

    left = ranks(np.asarray(values_a, dtype=np.float64))
    right = ranks(np.asarray(values_b, dtype=np.float64))
    if len(left) < 2:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left.dot(right) / denominator) if denominator else 0.0


def pearson(values_a: np.ndarray, values_b: np.ndarray) -> float:
    left = np.asarray(values_a, dtype=np.float64)
    right = np.asarray(values_b, dtype=np.float64)
    if len(left) < 2:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left.dot(right) / denominator) if denominator else 0.0


def load_validation(
    records: list[dict],
    processor,
    model,
    predictor,
    predictor_mean: np.ndarray,
    predictor_std: np.ndarray,
    device: torch.device,
    max_frames: int,
) -> list[dict]:
    rows = []
    for record in records:
        if record["split"] != "val":
            continue
        frames = list(
            iter_video_frames(
                PROJECT_ROOT / record["path"],
                max_frames=max_frames,
            )
        )
        previous_rgb = None
        previous_pixel = None
        full_logits_by_frame = {}
        features_by_frame = {}
        for frame_index, frame_rgb in frames:
            image = Image.fromarray(frame_rgb, mode="RGB")
            pixel_values = prepare_inputs(
                processor,
                image,
                device,
            )["pixel_values"]
            with torch.inference_mode():
                outputs = model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True,
                )
                features = get_stage_features(outputs, model=model)
                full_logits = model.decode_head(features).detach()
            full_logits_by_frame[frame_index] = full_logits
            features_by_frame[frame_index] = [value.detach() for value in features]
            if previous_rgb is not None:
                current = (
                    torch.from_numpy(frame_rgb)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .contiguous()
                    .to(device)
                )
                previous = (
                    torch.from_numpy(previous_rgb)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .contiguous()
                    .to(device)
                )
                scores_by_age = block_scores_by_age(
                    predictor,
                    predictor_mean,
                    predictor_std,
                    current,
                    previous,
                    max_age=max_frames,
                )
                block_scores = np.asarray(
                    scores_by_age["0"],
                    dtype=np.float32,
                )
                residual = gpu_rgb_residual(current, previous)
                observables = {
                    "rgb_residual_mean": float(residual.mean().item()),
                    "rgb_residual_p90": float(
                        torch.quantile(residual.float(), 0.90).item()
                    ),
                }
                current_stage = features[3]
                previous_stage = features_by_frame[frame_index - 1][3]
                stale_features = [*features[:3], previous_stage]
                with torch.inference_mode():
                    stale_logits = model.decode_head(stale_features)
                stale_metrics = _task_metrics(
                    full_logits,
                    stale_logits,
                    grid=4,
                )
                whole_stage_l2 = float(stale_metrics["logit_relative_l2"][0])
                whole_stage_kl = float(stale_metrics["global_kl"][0])
                whole_stage_pixel = float(
                    stale_metrics["global_pixel_disagreement"][0]
                )
                for block_index in range(16):
                    rows.append(
                        {
                            "video_id": record["video_id"],
                            "motion_level": record["motion_level"],
                            "frame": frame_index,
                            "frame_risk_scores": block_scores.tolist(),
                            "block_risks_by_age": json.dumps(scores_by_age),
                            "block_risk_max": float(block_scores.max()),
                            "block_risk_mean": float(block_scores.mean()),
                            "block_risk_p90": float(
                                np.quantile(block_scores, 0.90)
                            ),
                            "block_risk_topk_mean": float(
                                np.sort(block_scores)[-4:].mean()
                            ),
                            "task_damage": whole_stage_l2,
                            "global_kl": whole_stage_kl,
                            "pixel_disagreement": whole_stage_pixel,
                            "whole_stage_reuse_logit_l2": whole_stage_l2,
                            "whole_stage_reuse_kl": whole_stage_kl,
                            "whole_stage_reuse_pixel_disagreement": whole_stage_pixel,
                            "rgb_residual_mean": observables["rgb_residual_mean"],
                            "rgb_residual_p90": observables["rgb_residual_p90"],
                        }
                    )
            else:
                zero_scores = np.zeros(16, dtype=np.float32)
                zero_by_age = json.dumps(
                    {
                        str(age): zero_scores.tolist()
                        for age in range(max_frames + 1)
                    }
                )
                for block_index in range(16):
                    rows.append(
                        {
                            "video_id": record["video_id"],
                            "motion_level": record["motion_level"],
                            "frame": frame_index,
                            "frame_risk_scores": zero_scores.tolist(),
                            "block_risks_by_age": zero_by_age,
                            "block_risk_max": 0.0,
                            "block_risk_mean": 0.0,
                            "block_risk_p90": 0.0,
                            "block_risk_topk_mean": 0.0,
                            "task_damage": 0.0,
                            "global_kl": 0.0,
                            "pixel_disagreement": 0.0,
                            "whole_stage_reuse_logit_l2": 0.0,
                            "whole_stage_reuse_kl": 0.0,
                            "whole_stage_reuse_pixel_disagreement": 0.0,
                            "rgb_residual_mean": 0.0,
                            "rgb_residual_p90": 0.0,
                        }
                    )
            previous_rgb = frame_rgb
            previous_pixel = pixel_values
        del previous_pixel
    return rows


def choose_thresholds(frame_rows: list[dict]) -> dict[str, list[float]]:
    scores = {}
    for aggregation in AGGREGATIONS:
        values = []
        for row in frame_rows[::16]:
            by_age = json.loads(row["block_risks_by_age"])
            values.append(
                aggregate_frame_risk(
                    np.asarray(by_age["0"], dtype=np.float64),
                    aggregation,
                )
            )
        scores[aggregation] = np.asarray(values, dtype=np.float64)
    return {
        aggregation: [
            float(np.quantile(values, 1.0 - target))
            for target in TARGET_REFRESH_RATES
        ]
        for aggregation, values in scores.items()
    }


def run_policy(
    frame_rows: list[dict],
    policy,
    aggregation: str,
    threshold: float,
    max_age: int | None = None,
) -> tuple[list[dict], list[dict]]:
    del max_age
    # frame_rows has 16 intervention rows per frame; retain frame-level fields.
    frame_level = []
    for row in frame_rows[::16]:
        frame_level.append(row)
    by_video: dict[str, int] = {}
    result = []
    for row in frame_level:
        video_id = row["video_id"]
        age = by_video.get(video_id, 0)
        by_age = json.loads(row["block_risks_by_age"])
        risk_age = min(age, max(int(key) for key in by_age))
        risk = aggregate_frame_risk(
            np.asarray(by_age[str(risk_age)], dtype=np.float64),
            aggregation,
        )
        refresh = policy.decide(
            int(row["frame"]),
            age,
            frame_risk=risk,
        )
        if not by_video.get((video_id, "initialized"), False):
            refresh = True
            by_video[(video_id, "initialized")] = 1
        if refresh:
            age = 0
        else:
            age += 1
        by_video[video_id] = age
        result.append(
            {
                **row,
                "aggregation": aggregation,
                "threshold": threshold,
                "refresh": int(refresh),
                "stage_cache_age": age,
                "frame_risk": risk,
                "policy": type(policy).__name__,
            }
        )
    summary = {
        "policy": type(policy).__name__,
        "aggregation": aggregation,
        "threshold": threshold,
        "actual_refresh_rate": float(np.mean([row["refresh"] for row in result])),
        "mean_task_damage": float(
            np.mean(
                [
                    0.0
                    if row["refresh"]
                    else row["whole_stage_reuse_logit_l2"]
                    for row in result
                ]
            )
        ),
        "mean_kl": float(
            np.mean(
                [
                    0.0
                    if row["refresh"]
                    else row["whole_stage_reuse_kl"]
                    for row in result
                ]
            )
        ),
        "mean_pixel_disagreement": float(
            np.mean(
                [
                    0.0
                    if row["refresh"]
                    else row["whole_stage_reuse_pixel_disagreement"]
                    for row in result
                ]
            )
        ),
        "spearman_risk_damage": spearman(
            np.asarray([row["frame_risk"] for row in result]),
            np.asarray(
                [row["whole_stage_reuse_logit_l2"] for row in result]
            ),
        ),
        "pearson_risk_damage": pearson(
            np.asarray([row["frame_risk"] for row in result]),
            np.asarray(
                [row["whole_stage_reuse_logit_l2"] for row in result]
            ),
        ),
        "frames": len(result),
    }
    return result, [summary]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    records = read_csv(PROJECT_ROOT / args.inventory)
    processor, model = load_segformer(
        args.model,
        device=device,
        local_files_only=True,
    )
    predictor, predictor_mean, predictor_std = load_predictor(
        PROJECT_ROOT / args.predictor,
        device,
    )
    frame_rows = load_validation(
        records,
        processor,
        model,
        predictor,
        predictor_mean,
        predictor_std,
        device,
        args.max_frames,
    )
    frame_level = frame_rows[::16]
    aggregation_rows = []
    for aggregation in AGGREGATIONS:
        _, summary = run_policy(
            frame_rows,
            VisualRiskPolicy(threshold=float("inf")),
            aggregation,
            float("inf"),
        )
        aggregation_rows.extend(summary)
    thresholds_by_aggregation = choose_thresholds(frame_rows)
    threshold_rows = []
    for aggregation in AGGREGATIONS:
        for threshold in thresholds_by_aggregation[aggregation]:
            _, summary = run_policy(
                frame_rows,
                VisualRiskPolicy(threshold=threshold),
                aggregation,
                threshold,
            )
            threshold_rows.extend(summary)

    stage_age_visual_rows = []
    for aggregation in AGGREGATIONS:
        for threshold in thresholds_by_aggregation[aggregation]:
            for max_age in AMAX_VALUES:
                _, summary = run_policy(
                    frame_rows,
                    StageAgeVisualPolicy(
                        max_age=max_age,
                        threshold=threshold,
                    ),
                    aggregation,
                    threshold,
                    max_age=max_age,
                )
                summary[0]["max_age"] = max_age
                stage_age_visual_rows.extend(summary)

    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "frame_rows.csv", frame_rows)
    write_csv(output_root / "aggregation_validation.csv", aggregation_rows)
    write_csv(output_root / "threshold_validation.csv", threshold_rows)
    write_csv(output_root / "stage_age_visual_validation.csv", stage_age_visual_rows)
    (output_root / "config.yaml").write_text(
        "split: val\naggregations: [max, mean, p90, topk_mean]\n"
        "target_refresh_rates: [0.25, 0.40, 0.50, 0.60, 0.75]\n"
        "amax: [1, 2, 3, 5]\n"
        "predictor: frozen_patch_age\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    (output_root / "runtime.txt").write_text(
        "Validation-only visual aggregation and StageAgeVisual selection.\n",
        encoding="utf-8",
    )
    summary = {
        "validation_videos": sorted({row["video_id"] for row in frame_rows}),
        "aggregation_rows": aggregation_rows,
        "thresholds": thresholds_by_aggregation,
        "threshold_rows": threshold_rows,
        "stage_age_visual_rows": stage_age_visual_rows,
        "frame_rows": len(frame_rows),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
