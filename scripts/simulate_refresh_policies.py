"""Offline refresh-policy simulation using full current-frame features."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
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
    block_relative_l2,
    entropy_block_observables,
    replace_blocks,
    rgb_block_observables,
)
from driftguard.analysis.sequential_stale import decode_features  # noqa: E402
from driftguard.analysis.stage_sensitivity import stale_feature_metrics  # noqa: E402
from driftguard.models.segformer import (  # noqa: E402
    get_stage_features,
    load_segformer,
    prepare_inputs,
)
from driftguard.utils.common import choose_device, set_seed  # noqa: E402
from driftguard.utils.video import iter_video_frames  # noqa: E402


RATIOS = [0.10, 0.25, 0.40, 0.50, 0.75, 1.00]
POLICIES = ["random", "age", "rgb_residual", "oracle", "driftguard"]
BASE_FEATURE_COLUMNS = [
    "local_residual",
    "cache_age",
    "previous_entropy",
    "global_residual",
    "local_variance",
]


def _augment_features(values: np.ndarray, revision: str) -> np.ndarray:
    if revision == "baseline":
        return values
    if revision == "interactions":
        local = values[:, 0]
        age = values[:, 1]
        global_residual = values[:, 3]
        return np.column_stack(
            [
                values,
                local * age,
                global_residual * age,
                np.log1p(age),
            ]
        ).astype(np.float32)
    raise ValueError(f"Unknown feature revision: {revision}")


class TinyDriftEstimator(torch.nn.Module):
    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="data/real_videos")
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--estimator", default="results/E06_estimator/best_model.pt")
    parser.add_argument("--cut", type=int, default=None)
    parser.add_argument("--state", default="reports/research_state.json")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--block-grid", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-root", default="results/E07_refresh_simulation")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _read_cut(args: argparse.Namespace) -> int:
    if args.cut is not None:
        return args.cut
    state = json.loads((PROJECT_ROOT / args.state).read_text(encoding="utf-8"))
    value = state.get("selected_cache_cut")
    if value is None:
        raise ValueError("No selected cache cut in research_state.json")
    return int(value)


def _select_mask(scores: np.ndarray, ratio: float, rng: random.Random) -> np.ndarray:
    count = int(round(len(scores) * ratio))
    count = max(0, min(len(scores), count))
    mask = np.zeros(len(scores), dtype=bool)
    if count == len(scores):
        mask[:] = True
    elif count > 0:
        indices = np.argsort(scores)[-count:]
        mask[indices] = True
    return mask


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)
    cut = _read_cut(args)
    if cut < 1 or cut > 4:
        raise ValueError("--cut must be between 1 and 4")
    device = choose_device(args.device)
    video_dir = PROJECT_ROOT / args.video_dir
    videos = sorted(
        path
        for path in video_dir.glob("*")
        if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )
    if not videos:
        raise FileNotFoundError(f"No videos found in {video_dir}")

    checkpoint = torch.load(PROJECT_ROOT / args.estimator, map_location="cpu")
    estimator = TinyDriftEstimator(len(checkpoint["feature_columns"]))
    estimator.load_state_dict(checkpoint["state_dict"])
    estimator.to(device).eval()
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    std[std < 1e-8] = 1.0
    target_transform = checkpoint.get("target_transform", "identity")
    feature_revision = checkpoint.get("feature_revision", "baseline")

    processor, model = load_segformer(args.model, device=device)
    rows: list[dict] = []
    start_time = time.perf_counter()

    for video_path in videos:
        states = {
            (policy, ratio): {
                "cache": None,
                "age": np.zeros(args.block_grid * args.block_grid, dtype=np.float32),
            }
            for policy in POLICIES
            for ratio in RATIOS
        }
        previous_frame: np.ndarray | None = None
        previous_logits: torch.Tensor | None = None
        for frame_id, frame_rgb in iter_video_frames(
            video_path,
            max_frames=args.max_frames,
        ):
            image = Image.fromarray(frame_rgb, mode="RGB")
            inputs = prepare_inputs(processor, image, device)
            with torch.no_grad():
                outputs = model(
                    pixel_values=inputs["pixel_values"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                current_features = [
                    feature.detach()
                    for feature in get_stage_features(outputs, model=model)
                ]
                full_logits = decode_features(model, current_features)
                current_stage = current_features[cut - 1]
                cheap = rgb_block_observables(
                    frame_rgb,
                    previous_frame,
                    grid=args.block_grid,
                )
                entropy = entropy_block_observables(
                    previous_logits,
                    image_height=frame_rgb.shape[0],
                    image_width=frame_rgb.shape[1],
                    grid=args.block_grid,
                )

                for policy in POLICIES:
                    for ratio in RATIOS:
                        if policy == "full" and ratio != 1.0:
                            continue
                        state = states[(policy, ratio)]
                        if state["cache"] is None:
                            state["cache"] = current_stage.detach().clone()
                            state["age"][:] = 0.0
                        cache_before = state["cache"]
                        true_drift = block_relative_l2(
                            current_stage,
                            cache_before,
                            grid=args.block_grid,
                        )
                        features = np.column_stack(
                            [
                                cheap[:, 0],
                                state["age"],
                                entropy,
                                cheap[:, 2],
                                cheap[:, 3],
                            ]
                        ).astype(np.float32)
                        if policy == "random":
                            scores = np.asarray(
                                [rng.random() for _ in range(len(features))],
                                dtype=np.float32,
                            )
                        elif policy == "age":
                            scores = state["age"].copy()
                        elif policy == "rgb_residual":
                            scores = cheap[:, 0]
                        elif policy == "oracle":
                            scores = true_drift
                        elif policy == "driftguard":
                            model_features = _augment_features(features, feature_revision)
                            normalized = (model_features - mean) / std
                            scores = (
                                estimator(torch.from_numpy(normalized).to(device))
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            if target_transform == "log1p":
                                scores = np.expm1(scores)
                        else:
                            raise ValueError(policy)
                        refresh_mask = _select_mask(scores, ratio, rng)
                        state["cache"] = replace_blocks(
                            cache_before,
                            current_stage,
                            refresh_mask,
                            grid=args.block_grid,
                        )
                        state["age"] += 1.0
                        state["age"][refresh_mask] = 0.0
                        stale_features = list(current_features)
                        stale_features[cut - 1] = state["cache"]
                        metrics = stale_feature_metrics(
                            model=model,
                            current_features=current_features,
                            stale_features=stale_features,
                            full_logits=full_logits,
                        )
                        rows.append(
                            {
                                "video_id": video_path.stem,
                                "current_frame": frame_id,
                                "policy": policy,
                                "refresh_ratio": ratio,
                                "refresh_blocks": int(refresh_mask.sum()),
                                "pixel_agreement": 1.0
                                - metrics["pixel_disagreement_ratio"],
                                **metrics,
                            }
                        )
            previous_frame = frame_rgb
            previous_logits = full_logits.detach()

    output_root = PROJECT_ROOT / args.output_root
    _write_csv(output_root / "policy_metrics.csv", rows)
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["policy"], float(row["refresh_ratio"]))].append(row)
    summary = []
    for (policy, ratio), group in sorted(groups.items()):
        summary.append(
            {
                "policy": policy,
                "refresh_ratio": ratio,
                "samples": len(group),
                "pixel_agreement_mean": float(
                    np.mean([float(row["pixel_agreement"]) for row in group])
                ),
                "kl_divergence_mean": float(
                    np.mean([float(row["kl_divergence"]) for row in group])
                ),
                "pixel_disagreement_mean": float(
                    np.mean(
                        [float(row["pixel_disagreement_ratio"]) for row in group]
                    )
                ),
                "logit_relative_l2_mean": float(
                    np.mean([float(row["logit_relative_l2"]) for row in group])
                ),
            }
        )
    _write_csv(output_root / "summary.csv", summary)

    figures_dir = PROJECT_ROOT / "figures" / "E07_refresh_simulation"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename in (
        (
            "pixel_agreement_mean",
            "Pixel agreement to full inference",
            "refresh_ratio_vs_accuracy.png",
        ),
        (
            "logit_relative_l2_mean",
            "Logit relative L2 damage",
            "refresh_ratio_vs_output_damage.png",
        ),
    ):
        plt.figure(figsize=(8, 5))
        for policy in POLICIES:
            values = sorted(
                [row for row in summary if row["policy"] == policy],
                key=lambda row: row["refresh_ratio"],
            )
            if not values:
                continue
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
        plt.savefig(figures_dir / filename, dpi=160)
        plt.close()

    runtime = time.perf_counter() - start_time
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config.yaml").write_text(
        "\n".join(
            [
                f"video_dir: {args.video_dir}",
                f"model: {args.model}",
                f"estimator: {args.estimator}",
                f"cut: {cut}",
                f"max_frames: {args.max_frames}",
                f"block_grid: {args.block_grid}",
                f"device: {device}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output_root / "runtime.txt").write_text(f"{runtime:.6f}\n", encoding="utf-8")
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "cut": cut,
                "video_count": len(videos),
                "row_count": len(rows),
                "runtime_sec": runtime,
                "latency_interpretation": "offline output-damage simulation; not real partial GPU speedup",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"cut={cut}")
    print(f"videos={len(videos)}")
    print(f"rows={len(rows)}")
    print(f"runtime_sec={runtime:.3f}")
    print(f"output_root={output_root.resolve()}")


if __name__ == "__main__":
    main()
