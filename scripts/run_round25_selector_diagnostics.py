"""Run corrected deterministic selector diagnostics under sequential cache state."""

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
    entropy_block_observables,
    rgb_block_observables,
)
from driftguard.analysis.refresh_semantics import RefreshState, select_top_k  # noqa: E402
from driftguard.analysis.sequential_stale import decode_features  # noqa: E402
from driftguard.models.segformer import (  # noqa: E402
    get_stage_features,
    load_segformer,
    prepare_inputs,
)
from driftguard.utils.common import choose_device, set_seed  # noqa: E402
from driftguard.utils.video import iter_video_frames  # noqa: E402
from generate_round2_task_dataset import _patches, _task_metrics  # noqa: E402
from run_round25_selector_closure import (  # noqa: E402
    BLOCKS,
    GRID,
    RATIOS,
    _load_cnn,
    _load_old_mlp,
    _oracle_task_scores,
    _predict_cnn,
    _predict_old,
)


POLICIES = (
    "residual_mean",
    "residual_p90",
    "age",
    "old_tiny_mlp",
    "cnn_all",
    "cnn_patch_age",
    "oracle",
)
CORE_RATIOS = (0.25, 0.375, 0.5)


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
    parser.add_argument(
        "--inventory",
        default="results/R2_E00_video_dataset/video_inventory.csv",
    )
    parser.add_argument(
        "--model",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
    )
    parser.add_argument(
        "--old-mlp",
        default="results/R25_E01_selector_closure/old_tiny_mlp_taskrisk.pt",
    )
    parser.add_argument("--cnn-all", default="results/R2_E02_predictor/best_model.pt")
    parser.add_argument(
        "--cnn-patch-age",
        default="results/R2_E03_ablation/patch_age.pt",
    )
    parser.add_argument(
        "--output-root",
        default="results/R25_E02_selector_diagnostics",
    )
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--decoder-batch", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    old_mlp, old_mean, old_std = _load_old_mlp(
        PROJECT_ROOT / args.old_mlp, device
    )
    cnn_all, all_mean, all_std, all_indices = _load_cnn(
        PROJECT_ROOT / args.cnn_all, 5, device
    )
    cnn_pa, pa_mean, pa_std, pa_indices = _load_cnn(
        PROJECT_ROOT / args.cnn_patch_age, 1, device
    )
    with (PROJECT_ROOT / args.inventory).open(
        encoding="utf-8", newline=""
    ) as handle:
        records = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    processor, model = load_segformer(args.model, device=device)
    policy_rows: list[dict] = []
    block_rows: list[dict] = []
    start_time = time.perf_counter()

    for record in records:
        frames = list(
            iter_video_frames(
                PROJECT_ROOT / record["path"],
                max_frames=args.max_frames,
            )
        )
        states = {
            (policy, ratio): None
            for policy in POLICIES
            for ratio in RATIOS
        }
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
            updated_stages = []
            descriptors = []

            for ratio in RATIOS:
                score_by_policy = {}
                true_damage_by_policy = {}
                for policy in POLICIES:
                    state = states[(policy, ratio)]
                    metadata = base_metadata.copy()
                    metadata[:, 0] = state.age
                    if policy == "residual_mean":
                        score = cheap[:, 0]
                    elif policy == "residual_p90":
                        score = np.quantile(
                            patches.reshape(BLOCKS, -1), 0.90, axis=1
                        )
                    elif policy == "age":
                        score = state.age.copy()
                    elif policy == "old_tiny_mlp":
                        score = _predict_old(
                            old_mlp, metadata, old_mean, old_std, device
                        )
                    elif policy == "cnn_all":
                        score = _predict_cnn(
                            cnn_all,
                            patches,
                            metadata,
                            all_mean,
                            all_std,
                            all_indices,
                            device,
                        )
                    elif policy == "cnn_patch_age":
                        score = _predict_cnn(
                            cnn_pa,
                            patches,
                            metadata,
                            pa_mean,
                            pa_std,
                            pa_indices,
                            device,
                        )
                    elif policy == "oracle":
                        score = _oracle_task_scores(
                            model,
                            current_features,
                            state.cache,
                            full_logits,
                            args.decoder_batch,
                        )
                    else:
                        raise ValueError(policy)
                    score_by_policy[policy] = np.asarray(score, dtype=np.float32)

                    if ratio in CORE_RATIOS and policy != "oracle":
                        true_damage_by_policy[policy] = _oracle_task_scores(
                            model,
                            current_features,
                            state.cache,
                            full_logits,
                            args.decoder_batch,
                        )

                for policy in POLICIES:
                    state = states[(policy, ratio)]
                    score = score_by_policy[policy]
                    mask = select_top_k(score, ratio)
                    if ratio in CORE_RATIOS and policy != "oracle":
                        true_damage = true_damage_by_policy[policy]
                        for block_index in range(BLOCKS):
                            block_rows.append(
                                {
                                    "video_id": record["video_id"],
                                    "motion_level": record["motion_level"],
                                    "frame_id": frame_id,
                                    "policy": policy,
                                    "refresh_ratio": ratio,
                                    "block_index": block_index,
                                    "age_before": float(state.age[block_index]),
                                    "predictor_score": float(score[block_index]),
                                    "oracle_task_damage": float(
                                        true_damage[block_index]
                                    ),
                                    "selected": int(mask[block_index]),
                                }
                            )
                    state.apply_refresh(current_stage, mask, grid=GRID)
                    updated_stages.append(state.cache)
                    descriptors.append((policy, ratio, int(mask.sum())))

            for start in range(0, len(updated_stages), args.decoder_batch):
                batch_stage = torch.cat(
                    updated_stages[start : start + args.decoder_batch], dim=0
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
                    policy_rows.append(
                        {
                            "video_id": record["video_id"],
                            "motion_level": record["motion_level"],
                            "frame_id": frame_id,
                            "policy": policy,
                            "seed": 0,
                            "refresh_ratio": ratio,
                            "refresh_blocks": count,
                            "global_kl": float(metrics["global_kl"][index]),
                            "logit_relative_l2": float(
                                metrics["logit_relative_l2"][index]
                            ),
                            "global_pixel_disagreement": float(
                                metrics["global_pixel_disagreement"][index]
                            ),
                        }
                    )
            previous_frame = frame_rgb
            previous_logits = full_logits
        print(f"completed {record['video_id']}", flush=True)

    _write_csv(output_root / "corrected_policy_metrics.csv", policy_rows)
    _write_csv(output_root / "sequential_block_diagnostics.csv", block_rows)
    (output_root / "config.yaml").write_text(
        f"max_frames: {args.max_frames}\ndecoder_batch: {args.decoder_batch}\n"
        f"ratios: {list(RATIOS)}\ncore_diagnostic_ratios: {list(CORE_RATIOS)}\n"
        "oracle_reference: same_policy_pre_refresh_cache\n"
        f"device: {device}\n",
        encoding="utf-8",
    )
    (output_root / "runtime.txt").write_text(
        f"{time.perf_counter() - start_time:.6f}\n",
        encoding="utf-8",
    )
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "policy_rows": len(policy_rows),
                "block_rows": len(block_rows),
                "videos": [record["video_id"] for record in records],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"policy_rows": len(policy_rows), "block_rows": len(block_rows)}
        )
    )


if __name__ == "__main__":
    main()
