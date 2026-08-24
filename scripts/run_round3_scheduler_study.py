"""Run Round 3 age-aware sequential refresh scheduling experiments."""

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

from driftguard.analysis.age_aware_schedulers import (  # noqa: E402
    SchedulerDecision,
    select_afvt,
    select_magvr,
)
from driftguard.analysis.cheap_observables import rgb_block_observables  # noqa: E402
from driftguard.analysis.refresh_semantics import RefreshState, select_top_k  # noqa: E402
from driftguard.analysis.residual_patches import selected_residual_patches  # noqa: E402
from driftguard.analysis.sequential_stale import decode_features  # noqa: E402
from driftguard.models.segformer import (  # noqa: E402
    get_stage_features,
    load_segformer,
    prepare_inputs,
)
from driftguard.utils.common import choose_device, set_seed  # noqa: E402
from driftguard.utils.video import iter_video_frames  # noqa: E402
from generate_round2_task_dataset import _task_metrics  # noqa: E402
from run_round25_selector_closure import (  # noqa: E402
    BLOCKS,
    GRID,
    _load_cnn,
    _oracle_task_scores,
)


DEFAULT_RATIOS = (0.125, 0.25, 0.375, 0.5, 0.75, 1.0)
CORE_RATIOS = (0.25, 0.375, 0.5)
POLICIES = (
    "residual_mean",
    "age",
    "cnn_patch_age",
    "afvt",
    "magvr",
    "oracle",
)
OVERLAP_POLICIES = ("age", "cnn_patch_age", "afvt", "magvr")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _array_text(values: np.ndarray) -> str:
    return " ".join(str(int(value)) for value in values)


def _jain(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    denominator = len(values) * float(np.square(values).sum())
    if denominator == 0:
        return 1.0
    return float(values.sum() ** 2 / denominator)


def _visual_scores(
    model,
    metadata_mean: np.ndarray,
    metadata_std: np.ndarray,
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    ages: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    patches = selected_residual_patches(
        current_frame,
        previous_frame,
        indices,
        grid=GRID,
        size=16,
    )
    metadata = ages[indices, None].astype(np.float32)
    with torch.inference_mode():
        return (
            model(
                torch.from_numpy(patches).to(device),
                torch.from_numpy(
                    (metadata - metadata_mean) / metadata_std
                ).to(device),
            )
            .detach()
            .cpu()
            .numpy()
        )


def _blank_decision(mask: np.ndarray) -> SchedulerDecision:
    return SchedulerDecision(
        mask=mask,
        cnn_indices=np.empty(0, dtype=np.int64),
        tie_group_size=0,
        forced_blocks=0,
        visual_selected_blocks=0,
        forced_overflow=False,
        max_age=None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        default="results/R2_E00_video_dataset/video_inventory.csv",
    )
    parser.add_argument(
        "--model",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
    )
    parser.add_argument(
        "--predictor",
        default="results/R2_E03_ablation/patch_age.pt",
    )
    parser.add_argument(
        "--output-root",
        default="results/R3_E03_scheduler_comparison",
    )
    parser.add_argument(
        "--diagnostics-root",
        default="results/R3_E04_scheduler_diagnostics",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=list(DEFAULT_RATIOS),
    )
    parser.add_argument("--decoder-batch", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = tuple(sorted(set(float(value) for value in args.ratios)))
    if any(value < 0.0 or value > 1.0 for value in ratios):
        raise ValueError("refresh ratios must be between 0 and 1")
    set_seed(args.seed)
    device = choose_device(args.device)
    output_root = PROJECT_ROOT / args.output_root
    diagnostics_root = PROJECT_ROOT / args.diagnostics_root
    shard_root = output_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    diagnostics_root.mkdir(parents=True, exist_ok=True)

    predictor, metadata_mean, metadata_std, metadata_indices = _load_cnn(
        PROJECT_ROOT / args.predictor,
        metadata_dim=1,
        device=device,
    )
    if metadata_indices != [0]:
        raise ValueError("Round 3 requires the frozen Patch+Age checkpoint")
    with (PROJECT_ROOT / args.inventory).open(
        encoding="utf-8", newline=""
    ) as handle:
        records = [row for row in csv.DictReader(handle) if row["split"] == args.split]
    if args.max_videos is not None:
        records = records[: args.max_videos]
    processor, model = load_segformer(args.model, device=device)
    start_time = time.perf_counter()

    for record in records:
        prefix = shard_root / record["video_id"]
        metric_path = prefix.with_name(prefix.name + "_metrics.csv")
        frame_path = prefix.with_name(prefix.name + "_frames.csv")
        fairness_path = prefix.with_name(prefix.name + "_fairness.csv")
        overlap_path = prefix.with_name(prefix.name + "_overlap.csv")
        required = (metric_path, frame_path, fairness_path, overlap_path)
        if all(path.exists() for path in required) and not args.overwrite:
            print(f"skipped {record['video_id']} (checkpoint exists)", flush=True)
            continue

        frames = list(
            iter_video_frames(
                PROJECT_ROOT / record["path"],
                max_frames=args.max_frames,
            )
        )
        states = {
            (policy, ratio): None
            for policy in POLICIES
            for ratio in ratios
        }
        refresh_counts = {
            key: np.zeros(BLOCKS, dtype=np.int32) for key in states
        }
        max_stale_runs = {
            key: np.zeros(BLOCKS, dtype=np.int32) for key in states
        }
        metric_rows: list[dict] = []
        frame_rows: list[dict] = []
        overlap_rows: list[dict] = []
        previous_frame = None

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
                continue

            cheap = rgb_block_observables(frame_rgb, previous_frame, grid=GRID)
            updated_stages = []
            descriptors = []
            for ratio in ratios:
                for policy in POLICIES:
                    key = (policy, ratio)
                    state = states[key]
                    ages_before = state.age.copy()

                    def visual_score(indices, ages=ages_before):
                        return _visual_scores(
                            predictor,
                            metadata_mean,
                            metadata_std,
                            frame_rgb,
                            previous_frame,
                            ages,
                            indices,
                            device,
                        )

                    if policy == "residual_mean":
                        decision = _blank_decision(
                            select_top_k(cheap[:, 0], ratio)
                        )
                    elif policy == "age":
                        decision = _blank_decision(
                            select_top_k(ages_before, ratio)
                        )
                    elif policy == "cnn_patch_age":
                        all_indices = np.arange(BLOCKS, dtype=np.int64)
                        scores = visual_score(all_indices)
                        decision = SchedulerDecision(
                            mask=select_top_k(scores, ratio),
                            cnn_indices=all_indices,
                            tie_group_size=0,
                            forced_blocks=0,
                            visual_selected_blocks=int(
                                select_top_k(scores, ratio).sum()
                            ),
                            forced_overflow=False,
                            max_age=None,
                        )
                    elif policy == "afvt":
                        decision = select_afvt(
                            ages_before,
                            ratio,
                            visual_score,
                        )
                    elif policy == "magvr":
                        decision = select_magvr(
                            ages_before,
                            ratio,
                            visual_score,
                        )
                    elif policy == "oracle":
                        scores = _oracle_task_scores(
                            model,
                            current_features,
                            state.cache,
                            full_logits,
                            args.decoder_batch,
                        )
                        decision = _blank_decision(
                            select_top_k(scores, ratio)
                        )
                    else:
                        raise ValueError(policy)

                    if ratio in CORE_RATIOS and policy in OVERLAP_POLICIES:
                        true_damage = _oracle_task_scores(
                            model,
                            current_features,
                            state.cache,
                            full_logits,
                            args.decoder_batch,
                        )
                        oracle_mask = select_top_k(true_damage, ratio)
                        count = int(decision.mask.sum())
                        overlap_rows.append(
                            {
                                "video_id": record["video_id"],
                                "motion_group": record["motion_level"],
                                "frame_id": frame_id,
                                "method": policy,
                                "budget": ratio,
                                "selected_blocks": count,
                                "oracle_overlap": float(
                                    np.logical_and(
                                        decision.mask, oracle_mask
                                    ).sum()
                                    / max(1, count)
                                ),
                            }
                        )

                    state.apply_refresh(
                        current_stage,
                        decision.mask,
                        grid=GRID,
                    )
                    refresh_counts[key] += decision.mask.astype(np.int32)
                    max_stale_runs[key] = np.maximum(
                        max_stale_runs[key],
                        state.age.astype(np.int32),
                    )
                    ages_after = state.age.copy()
                    frame_rows.append(
                        {
                            "video_id": record["video_id"],
                            "motion_group": record["motion_level"],
                            "frame_id": frame_id,
                            "method": policy,
                            "budget": ratio,
                            "ages_before": _array_text(ages_before),
                            "ages_after": _array_text(ages_after),
                            "mean_age": float(ages_after.mean()),
                            "median_age": float(np.median(ages_after)),
                            "p90_age": float(np.quantile(ages_after, 0.90)),
                            "max_age": float(ages_after.max()),
                            "age_variance": float(ages_after.var()),
                            "age_0_blocks": int((ages_after == 0).sum()),
                            "age_1_blocks": int((ages_after == 1).sum()),
                            "age_2_blocks": int((ages_after == 2).sum()),
                            "age_3_blocks": int((ages_after == 3).sum()),
                            "age_4plus_blocks": int((ages_after >= 4).sum()),
                            "selected_indices": _array_text(
                                np.flatnonzero(decision.mask)
                            ),
                            "tie_group_size": decision.tie_group_size,
                            "cnn_evaluated_blocks": decision.cnn_evaluated_blocks,
                            "cnn_calls": decision.cnn_calls,
                            "forced_blocks": decision.forced_blocks,
                            "visual_selected_blocks": (
                                decision.visual_selected_blocks
                            ),
                            "forced_overflow": int(
                                decision.forced_overflow
                            ),
                            "budget_max_age": (
                                ""
                                if decision.max_age is None
                                else decision.max_age
                            ),
                        }
                    )
                    updated_stages.append(state.cache)
                    descriptors.append(
                        (policy, ratio, int(decision.mask.sum()))
                    )

            for start in range(0, len(updated_stages), args.decoder_batch):
                batch_stage = torch.cat(
                    updated_stages[start : start + args.decoder_batch],
                    dim=0,
                )
                batch_features = [
                    feature.expand(batch_stage.shape[0], -1, -1, -1)
                    for feature in current_features
                ]
                batch_features[3] = batch_stage
                with torch.inference_mode():
                    stale_logits = decode_features(model, batch_features)
                metrics = _task_metrics(full_logits, stale_logits, GRID)
                for index, (policy, ratio, count) in enumerate(
                    descriptors[start : start + args.decoder_batch]
                ):
                    metric_rows.append(
                        {
                            "video_id": record["video_id"],
                            "motion_group": record["motion_level"],
                            "frame_id": frame_id,
                            "method": policy,
                            "budget": ratio,
                            "refresh_blocks": count,
                            "logit_relative_l2": float(
                                metrics["logit_relative_l2"][index]
                            ),
                            "global_kl": float(metrics["global_kl"][index]),
                            "pixel_disagreement": float(
                                metrics["global_pixel_disagreement"][index]
                            ),
                        }
                    )
            previous_frame = frame_rgb

        fairness_rows = []
        for (policy, ratio), counts in refresh_counts.items():
            stale = max_stale_runs[(policy, ratio)]
            count_mean = float(counts.mean())
            fairness_rows.append(
                {
                    "video_id": record["video_id"],
                    "motion_group": record["motion_level"],
                    "method": policy,
                    "budget": ratio,
                    "refresh_counts": _array_text(counts),
                    "max_stale_runs": _array_text(stale),
                    "mean_max_stale_run": float(stale.mean()),
                    "p95_max_stale_run": float(
                        np.quantile(stale, 0.95)
                    ),
                    "max_stale_run": int(stale.max()),
                    "refresh_count_std": float(counts.std()),
                    "refresh_count_cv": float(
                        counts.std() / count_mean if count_mean > 0 else 0.0
                    ),
                    "jain_fairness": _jain(counts),
                }
            )
        _write_csv(metric_path, metric_rows)
        _write_csv(frame_path, frame_rows)
        _write_csv(fairness_path, fairness_rows)
        _write_csv(overlap_path, overlap_rows)
        print(f"completed {record['video_id']}", flush=True)

    all_metrics = []
    all_frames = []
    all_fairness = []
    all_overlap = []
    for record in records:
        prefix = shard_root / record["video_id"]
        all_metrics.extend(
            _read_csv(prefix.with_name(prefix.name + "_metrics.csv"))
        )
        all_frames.extend(
            _read_csv(prefix.with_name(prefix.name + "_frames.csv"))
        )
        all_fairness.extend(
            _read_csv(prefix.with_name(prefix.name + "_fairness.csv"))
        )
        all_overlap.extend(
            _read_csv(prefix.with_name(prefix.name + "_overlap.csv"))
        )
    _write_csv(output_root / "metrics.csv", all_metrics)
    _write_csv(diagnostics_root / "scheduler_frame_diagnostics.csv", all_frames)
    _write_csv(diagnostics_root / "cache_fairness_per_video.csv", all_fairness)
    _write_csv(diagnostics_root / "oracle_overlap_per_frame.csv", all_overlap)

    summary = []
    for policy in POLICIES:
        for ratio in ratios:
            values = [
                row
                for row in all_metrics
                if row["method"] == policy
                and float(row["budget"]) == ratio
            ]
            if not values:
                continue
            summary.append(
                {
                    "method": policy,
                    "budget": ratio,
                    "samples": len(values),
                    "video_count": len(
                        {row["video_id"] for row in values}
                    ),
                    "logit_relative_l2_mean": float(
                        np.mean(
                            [
                                float(row["logit_relative_l2"])
                                for row in values
                            ]
                        )
                    ),
                    "global_kl_mean": float(
                        np.mean([float(row["global_kl"]) for row in values])
                    ),
                    "pixel_disagreement_mean": float(
                        np.mean(
                            [
                                float(row["pixel_disagreement"])
                                for row in values
                            ]
                        )
                    ),
                }
            )
    _write_csv(output_root / "summary.csv", summary)
    config_text = (
        f"split: {args.split}\nmax_videos: {args.max_videos}\n"
        f"max_frames: {args.max_frames}\nratios: {list(ratios)}\n"
        f"decoder_batch: {args.decoder_batch}\ndevice: {device}\n"
        f"seed: {args.seed}\npredictor: {args.predictor}\n"
        "magvr_max_age: budget_derived\n"
    )
    for root in (output_root, diagnostics_root):
        (root / "config.yaml").write_text(config_text, encoding="utf-8")
        (root / "command.txt").write_text(
            " ".join(sys.argv) + "\n", encoding="utf-8"
        )
        (root / "runtime.txt").write_text(
            f"{time.perf_counter() - start_time:.6f}\n",
            encoding="utf-8",
        )
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "videos": [record["video_id"] for record in records],
                "metrics_rows": len(all_metrics),
                "frame_diagnostic_rows": len(all_frames),
                "summary": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "videos": len(records),
                "metrics_rows": len(all_metrics),
            }
        )
    )


if __name__ == "__main__":
    main()
