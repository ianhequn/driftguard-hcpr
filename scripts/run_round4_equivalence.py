"""Verify frozen Round 3 AFVT against the Round 4 GPU implementation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from driftguard.analysis.age_aware_schedulers import select_afvt  # noqa: E402
from driftguard.analysis.refresh_semantics import RefreshState  # noqa: E402
from driftguard.analysis.residual_patches import selected_residual_patches  # noqa: E402
from driftguard.analysis.sequential_stale import decode_features  # noqa: E402
from driftguard.models.risk_predictor import TinyRiskCNN  # noqa: E402
from driftguard.models.segformer import (  # noqa: E402
    get_stage_features,
    load_segformer,
    prepare_inputs,
)
from driftguard.profiling.selector_gpu import GPUAFVTSelector  # noqa: E402
from driftguard.utils.video import iter_video_frames  # noqa: E402
from generate_round2_task_dataset import _task_metrics  # noqa: E402


CORE_RATIOS = (0.25, 0.375, 0.5)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_predictor(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    model = TinyRiskCNN(metadata_dim=1)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    mean = np.asarray(checkpoint["metadata_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["metadata_std"], dtype=np.float32)
    std[std < 1e-8] = 1.0
    return model, mean, std


def _frame_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(frame)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(device)
    )


def _indices_text(mask: np.ndarray) -> str:
    return " ".join(str(value) for value in np.flatnonzero(mask))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        default="results/R2_E00_video_dataset/video_inventory.csv",
    )
    parser.add_argument(
        "--round3-diagnostics",
        default=(
            "results/R3_E04_scheduler_diagnostics/"
            "scheduler_frame_diagnostics.csv"
        ),
    )
    parser.add_argument(
        "--round3-summary",
        default="results/R3_E03_scheduler_comparison/secondary_metrics.csv",
    )
    parser.add_argument(
        "--model",
        default="/root/autodl-tmp/models/segformer-b0",
    )
    parser.add_argument(
        "--predictor",
        default="results/R2_E03_ablation/patch_age.pt",
    )
    parser.add_argument(
        "--output-root",
        default="results/R4_E06_equivalence",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    predictor, mean, std = _load_predictor(
        PROJECT_ROOT / args.predictor, device
    )
    selector = GPUAFVTSelector(
        predictor,
        mean,
        std,
        extraction="all_blocks",
        fast_path=True,
    )
    processor, segmentation = load_segformer(args.model, device=device)
    records = [
        row
        for row in _read_csv(PROJECT_ROOT / args.inventory)
        if row["split"] == args.split
    ]
    stored = {
        (
            row["video_id"],
            int(row["frame_id"]),
            float(row["budget"]),
        ): row
        for row in _read_csv(PROJECT_ROOT / args.round3_diagnostics)
        if row["method"] == "afvt"
        and float(row["budget"]) in CORE_RATIOS
    }
    stored_video_ids = {key[0] for key in stored}
    records = [
        row for row in records if row["video_id"] in stored_video_ids
    ]
    if args.max_videos is not None:
        records = records[: args.max_videos]
    if not records:
        raise RuntimeError(
            "No inventory videos match the frozen Round 3 diagnostics"
        )

    equivalence_rows = []
    metric_rows = []
    fairness_rows = []
    for record in records:
        states = {ratio: None for ratio in CORE_RATIOS}
        max_stale = {
            ratio: np.zeros(16, dtype=np.int32) for ratio in CORE_RATIOS
        }
        frames = list(
            iter_video_frames(
                PROJECT_ROOT / record["path"],
                max_frames=args.max_frames,
            )
        )
        previous = None
        for frame_id, frame in frames:
            image = Image.fromarray(frame, mode="RGB")
            inputs = prepare_inputs(processor, image, device)
            with torch.inference_mode():
                outputs = segmentation(
                    pixel_values=inputs["pixel_values"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                features = [
                    value.detach()
                    for value in get_stage_features(
                        outputs, model=segmentation
                    )
                ]
                full_logits = decode_features(
                    segmentation, features
                ).detach()
            current_stage = features[3]
            if previous is None:
                for ratio in CORE_RATIOS:
                    states[ratio] = RefreshState.initialize(
                        current_stage, 16
                    )
                previous = frame
                continue

            current_gpu = _frame_tensor(frame, device)
            previous_gpu = _frame_tensor(previous, device)
            updated = []
            descriptors = []
            for ratio in CORE_RATIOS:
                state = states[ratio]
                ages = state.age.copy()
                old_scores = {}

                def old_score(indices):
                    patches = selected_residual_patches(
                        frame, previous, indices, grid=4, size=16
                    )
                    metadata = (
                        (ages[indices, None] - mean) / std
                    ).astype(np.float32)
                    with torch.inference_mode():
                        scores = (
                            predictor(
                                torch.from_numpy(patches).to(device),
                                torch.from_numpy(metadata).to(device),
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    old_scores["indices"] = indices.copy()
                    old_scores["scores"] = scores.copy()
                    return scores

                old = select_afvt(ages, ratio, old_score)
                new = selector.select(
                    current_gpu, previous_gpu, ages, ratio
                )
                key = (record["video_id"], frame_id, ratio)
                stored_indices = stored[key]["selected_indices"]
                max_diff = 0.0
                mean_diff = 0.0
                if new.used_visual_path:
                    old_values = old_scores["scores"]
                    difference = np.abs(
                        old_values - new.risk_scores
                    )
                    max_diff = float(difference.max())
                    mean_diff = float(difference.mean())
                equivalence_rows.append(
                    {
                        "video_id": record["video_id"],
                        "frame_id": frame_id,
                        "budget": ratio,
                        "old_selected_blocks": _indices_text(old.mask),
                        "new_selected_blocks": _indices_text(
                            new.decision.mask
                        ),
                        "stored_selected_blocks": stored_indices,
                        "exact_match": int(
                            np.array_equal(
                                old.mask, new.decision.mask
                            )
                            and _indices_text(new.decision.mask)
                            == stored_indices
                        ),
                        "max_risk_diff": max_diff,
                        "mean_risk_diff": mean_diff,
                        "cnn_evaluated_blocks": (
                            new.decision.cnn_evaluated_blocks
                        ),
                    }
                )
                state.apply_refresh(
                    current_stage, new.decision.mask, grid=4
                )
                max_stale[ratio] = np.maximum(
                    max_stale[ratio], state.age.astype(np.int32)
                )
                updated.append(state.cache)
                descriptors.append(ratio)

            batch_stage = torch.cat(updated, dim=0)
            batch_features = [
                value.expand(len(updated), -1, -1, -1)
                for value in features
            ]
            batch_features[3] = batch_stage
            with torch.inference_mode():
                stale_logits = decode_features(
                    segmentation, batch_features
                )
            metrics = _task_metrics(full_logits, stale_logits, 4)
            for index, ratio in enumerate(descriptors):
                metric_rows.append(
                    {
                        "video_id": record["video_id"],
                        "motion_group": record["motion_level"],
                        "frame_id": frame_id,
                        "budget": ratio,
                        "logit_relative_l2": float(
                            metrics["logit_relative_l2"][index]
                        ),
                        "global_kl": float(
                            metrics["global_kl"][index]
                        ),
                        "pixel_disagreement": float(
                            metrics[
                                "global_pixel_disagreement"
                            ][index]
                        ),
                    }
                )
            previous = frame

        for ratio in CORE_RATIOS:
            fairness_rows.append(
                {
                    "video_id": record["video_id"],
                    "budget": ratio,
                    "max_stale_run": int(max_stale[ratio].max()),
                    "max_stale_runs": " ".join(
                        str(value) for value in max_stale[ratio]
                    ),
                }
            )
        print(f"completed {record['video_id']}", flush=True)

    summary_rows = []
    round3 = {}
    if args.split == "test":
        round3 = {
            (row["method"], float(row["budget"])): row
            for row in _read_csv(PROJECT_ROOT / args.round3_summary)
        }
    for ratio in CORE_RATIOS:
        values = [
            row for row in metric_rows if row["budget"] == ratio
        ]
        source = round3.get(("afvt", ratio))
        summary_rows.append(
            {
                "Budget": ratio,
                "Round4LogitRelativeL2": float(
                    np.mean(
                        [row["logit_relative_l2"] for row in values]
                    )
                ),
                "Round3LogitRelativeL2": (
                    ""
                    if source is None
                    else float(source["logit_relative_l2"])
                ),
                "Round4KL": float(
                    np.mean([row["global_kl"] for row in values])
                ),
                "Round3KL": (
                    "" if source is None else float(source["global_kl"])
                ),
                "Round4PixelDisagreement": float(
                    np.mean(
                        [row["pixel_disagreement"] for row in values]
                    )
                ),
                "Round3PixelDisagreement": (
                    ""
                    if source is None
                    else float(source["pixel_disagreement"])
                ),
                "MaxStaleRun": max(
                    row["max_stale_run"]
                    for row in fairness_rows
                    if row["budget"] == ratio
                ),
            }
        )

    output_root = PROJECT_ROOT / args.output_root
    _write_csv(output_root / "policy_equivalence.csv", equivalence_rows)
    _write_csv(output_root / "task_metric_equivalence.csv", summary_rows)
    _write_csv(output_root / "fairness_equivalence.csv", fairness_rows)
    exact = float(
        np.mean([row["exact_match"] for row in equivalence_rows])
    )
    summary = {
        "split": args.split,
        "videos": [row["video_id"] for row in records],
        "policy_rows": len(equivalence_rows),
        "exact_match_rate": exact,
        "max_risk_diff": max(
            row["max_risk_diff"] for row in equivalence_rows
        ),
        "mean_risk_diff": float(
            np.mean(
                [row["mean_risk_diff"] for row in equivalence_rows]
            )
        ),
        "task_metrics": summary_rows,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "config.yaml").write_text(
        f"split: {args.split}\nmax_videos: {args.max_videos}\n"
        f"max_frames: {args.max_frames}\nprecision: fp32\n"
        "reference: immediately_previous_rgb_frame\n"
        "extraction: all_blocks\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(
        " ".join(sys.argv) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
