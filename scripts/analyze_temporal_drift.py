"""Characterize feature drift versus cache age and task damage."""

from __future__ import annotations

import argparse
import csv
import json
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
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.analysis.sequential_stale import (  # noqa: E402
    decode_features,
    sequential_stale_features,
)
from driftguard.analysis.stage_sensitivity import stale_feature_metrics  # noqa: E402
from driftguard.analysis.temporal_similarity import relative_l2_distance  # noqa: E402
from driftguard.models.segformer import (  # noqa: E402
    get_stage_features,
    load_segformer,
    prepare_inputs,
)
from driftguard.utils.common import choose_device, set_seed  # noqa: E402
from driftguard.utils.video import iter_video_frames  # noqa: E402


FIELDS = [
    "video_id",
    "current_frame",
    "reference_frame",
    "cache_age",
    "motion_residual",
    "feature_drift",
    "kl_divergence",
    "pixel_disagreement_ratio",
    "logit_relative_l2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="data/real_videos")
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--cut", type=int, default=None)
    parser.add_argument("--state", default="reports/research_state.json")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--ages", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-root", default="results/E04_temporal_drift")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_cut(args: argparse.Namespace) -> int:
    if args.cut is not None:
        return args.cut
    state = json.loads((PROJECT_ROOT / args.state).read_text(encoding="utf-8"))
    cut = state.get("selected_cache_cut")
    if cut is None:
        raise ValueError("No selected cache cut in research_state.json")
    return int(cut)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cut = _read_cut(args)
    if cut < 1 or cut > 4:
        raise ValueError("--cut must be between 1 and 4")
    ages = sorted({int(value) for value in args.ages})
    if not ages or any(value <= 0 for value in ages):
        raise ValueError("--ages must contain positive integers")
    device = choose_device(args.device)
    video_dir = PROJECT_ROOT / args.video_dir
    videos = sorted(
        path
        for path in video_dir.glob("*")
        if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )
    if not videos:
        raise FileNotFoundError(f"No videos found in {video_dir}")

    processor, model = load_segformer(args.model, device=device)
    rows: list[dict] = []
    start_time = time.perf_counter()
    for video_path in videos:
        history: dict[int, list[torch.Tensor]] = {}
        previous_frame: np.ndarray | None = None
        max_age = max(ages)
        for frame_id, frame_rgb in iter_video_frames(
            video_path,
            max_frames=args.max_frames,
        ):
            if previous_frame is None:
                motion = 0.0
            else:
                motion = float(
                    np.abs(
                        frame_rgb.astype(np.float32)
                        - previous_frame.astype(np.float32)
                    ).mean()
                    / 255.0
                )
            previous_frame = frame_rgb
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
                for age in ages:
                    reference_frame = frame_id - age
                    cached = history.get(reference_frame)
                    if cached is None:
                        continue
                    stale = sequential_stale_features(
                        model=model,
                        current_features=current_features,
                        cached_features=cached,
                        cut=cut,
                    )
                    metrics = stale_feature_metrics(
                        model=model,
                        current_features=current_features,
                        stale_features=stale,
                        full_logits=full_logits,
                    )
                    rows.append(
                        {
                            "video_id": video_path.stem,
                            "current_frame": frame_id,
                            "reference_frame": reference_frame,
                            "cache_age": age,
                            "motion_residual": motion,
                            "feature_drift": relative_l2_distance(
                                current_features[cut - 1],
                                cached[cut - 1],
                            ),
                            **metrics,
                        }
                    )
            history[frame_id] = [feature.detach() for feature in current_features]
            for old_frame in list(history):
                if old_frame < frame_id - max_age:
                    del history[old_frame]

    output_root = PROJECT_ROOT / args.output_root
    _write_csv(output_root / "drift_vs_age.csv", FIELDS, rows)

    by_age: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_age[int(row["cache_age"])].append(row)
    distribution_rows = []
    for age, group in sorted(by_age.items()):
        values = np.asarray([float(row["feature_drift"]) for row in group])
        distribution_rows.append(
            {
                "cache_age": age,
                "num_samples": len(group),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std()),
                "p75": float(np.quantile(values, 0.75)),
                "p90": float(np.quantile(values, 0.90)),
                "p95": float(np.quantile(values, 0.95)),
            }
        )
    _write_csv(
        output_root / "drift_distribution.csv",
        list(distribution_rows[0]),
        distribution_rows,
    )

    correlation_rows = []
    targets = {
        "kl_divergence": "kl_divergence",
        "pixel_disagreement_ratio": "pixel_disagreement_ratio",
        "logit_relative_l2": "logit_relative_l2",
        "motion_residual": "motion_residual",
    }
    feature_values = np.asarray([float(row["feature_drift"]) for row in rows])
    for name, field in targets.items():
        target_values = np.asarray([float(row[field]) for row in rows])
        correlation_rows.append(
            {
                "feature": "feature_drift",
                "target": name,
                "num_samples": len(rows),
                "pearson": float(pearsonr(feature_values, target_values).statistic),
                "spearman": float(spearmanr(feature_values, target_values).statistic),
            }
        )
    _write_csv(
        output_root / "feature_task_correlation.csv",
        list(correlation_rows[0]),
        correlation_rows,
    )

    figures_dir = PROJECT_ROOT / "figures" / "E04_temporal_drift"
    figures_dir.mkdir(parents=True, exist_ok=True)
    ages_sorted = sorted(by_age)
    means = [
        float(np.mean([float(row["feature_drift"]) for row in by_age[age]]))
        for age in ages_sorted
    ]
    plt.figure(figsize=(7, 5))
    plt.plot(ages_sorted, means, marker="o")
    plt.xlabel("Cache age")
    plt.ylabel("Mean feature drift")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "drift_vs_cache_age.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(
        feature_values,
        [float(row["logit_relative_l2"]) for row in rows],
        s=8,
        alpha=0.35,
    )
    plt.xlabel("Feature drift")
    plt.ylabel("Task logit relative L2")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_drift_vs_task_damage.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    for age in ages_sorted:
        values = [float(row["feature_drift"]) for row in by_age[age]]
        plt.hist(values, bins=30, alpha=0.35, label=f"age {age}")
    plt.xlabel("Feature drift")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "drift_distribution_by_motion.png", dpi=160)
    plt.close()

    runtime = time.perf_counter() - start_time
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config.yaml").write_text(
        "\n".join(
            [
                f"video_dir: {args.video_dir}",
                f"model: {args.model}",
                f"cut: {cut}",
                f"max_frames: {args.max_frames}",
                f"ages: {ages}",
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
