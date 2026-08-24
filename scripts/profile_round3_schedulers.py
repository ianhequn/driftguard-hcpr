"""Profile Round 3 schedulers with their actual conditional visual inputs."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.analysis.age_aware_schedulers import (  # noqa: E402
    select_afvt,
    select_magvr,
)
from driftguard.analysis.refresh_semantics import select_top_k  # noqa: E402
from driftguard.analysis.residual_patches import selected_residual_patches  # noqa: E402
from driftguard.models.risk_predictor import TinyRiskCNN  # noqa: E402
from driftguard.utils.common import choose_device  # noqa: E402
from driftguard.utils.video import iter_video_frames  # noqa: E402


CORE_RATIOS = (0.25, 0.375, 0.5)
METHODS = ("Age", "Pure CNN", "AFVT", "MAG-VR")
METHOD_KEYS = {
    "Age": "age",
    "Pure CNN": "cnn_patch_age",
    "AFVT": "afvt",
    "MAG-VR": "magvr",
}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(array.mean()),
        float(np.median(array)),
        float(np.quantile(array, 0.95)),
        float(np.quantile(array, 0.99)),
    )


def _measure_wall(fn, samples, warmup, repeats, device):
    for index in range(warmup):
        fn(samples[index % len(samples)])
    timings = []
    for index in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn(samples[index % len(samples)])
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return _stats(timings)


def _prime_cuda(device: torch.device) -> None:
    if device.type != "cuda":
        return
    values = torch.randn((2048, 2048), device=device)
    for _ in range(20):
        values = values @ values
    torch.cuda.synchronize()


def _measure_model(fn, prepared, warmup, repeats, device):
    active = [value for value in prepared if value[0].shape[0] > 0]
    if not active:
        return (0.0, 0.0, 0.0, 0.0)
    for index in range(warmup):
        fn(active[index % len(active)])
    _prime_cuda(device)
    timings = []
    active_index = 0
    for index in range(repeats):
        value = prepared[index % len(prepared)]
        if value[0].shape[0] == 0:
            timings.append(0.0)
            continue
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(value)
            end.record()
            torch.cuda.synchronize()
            timings.append(float(start.elapsed_time(end)))
        else:
            start_time = time.perf_counter()
            fn(value)
            timings.append((time.perf_counter() - start_time) * 1000.0)
        active_index += 1
    return _stats(timings)


def _parse_ages(text: str) -> np.ndarray:
    return np.asarray([float(value) for value in text.split()], dtype=np.float32)


def _load_samples(inventory_path: Path, diagnostics_path: Path):
    inventory = {
        row["video_id"]: row
        for row in _read_csv(inventory_path)
        if row["split"] == "test"
    }
    frame_maps = {}
    for video_id, record in inventory.items():
        frame_maps[video_id] = {
            frame_id: frame
            for frame_id, frame in iter_video_frames(
                PROJECT_ROOT / record["path"],
                max_frames=48,
            )
        }
    rows = _read_csv(diagnostics_path)
    samples = {}
    for method in METHODS:
        key = METHOD_KEYS[method]
        for ratio in CORE_RATIOS:
            selected = [
                row
                for row in rows
                if row["method"] == key
                and float(row["budget"]) == ratio
            ]
            values = []
            for row in selected:
                frame_id = int(row["frame_id"])
                frames = frame_maps[row["video_id"]]
                values.append(
                    {
                        "current": frames[frame_id],
                        "previous": frames[frame_id - 1],
                        "ages": _parse_ages(row["ages_before"]),
                    }
                )
            samples[(method, ratio)] = values
    return samples


def _load_predictor(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    model = TinyRiskCNN(metadata_dim=1)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    mean = np.asarray(checkpoint["metadata_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["metadata_std"], dtype=np.float32)
    std[std < 1e-8] = 1.0
    return model, mean, std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        default="results/R2_E00_video_dataset/video_inventory.csv",
    )
    parser.add_argument(
        "--diagnostics",
        default=(
            "results/R3_E04_scheduler_diagnostics/"
            "scheduler_frame_diagnostics.csv"
        ),
    )
    parser.add_argument(
        "--predictor",
        default="results/R2_E03_ablation/patch_age.pt",
    )
    parser.add_argument(
        "--output-root",
        default="results/R3_E05_scheduler_profiling",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 100 or args.repeats < 500:
        raise ValueError("Round 3 profiling requires warmup>=100 and repeats>=500")
    device = choose_device(args.device)
    model, mean, std = _load_predictor(PROJECT_ROOT / args.predictor, device)
    samples = _load_samples(
        PROJECT_ROOT / args.inventory,
        PROJECT_ROOT / args.diagnostics,
    )
    rows = []

    for method in METHODS:
        for ratio in CORE_RATIOS:
            method_samples = samples[(method, ratio)]

            def candidate_indices(sample):
                if method == "Pure CNN":
                    return np.arange(16, dtype=np.int64)
                if method == "Age":
                    return np.empty(0, dtype=np.int64)
                captured = {}

                def score(indices):
                    captured["indices"] = np.asarray(indices, dtype=np.int64)
                    return np.zeros(len(indices), dtype=np.float32)

                scheduler = select_afvt if method == "AFVT" else select_magvr
                scheduler(sample["ages"], ratio, score)
                return captured.get(
                    "indices", np.empty(0, dtype=np.int64)
                )

            def prepare(sample):
                indices = candidate_indices(sample)
                patches = selected_residual_patches(
                    sample["current"],
                    sample["previous"],
                    indices,
                    grid=4,
                    size=16,
                )
                metadata = sample["ages"][indices, None].astype(np.float32)
                return (
                    torch.from_numpy(patches).to(device),
                    torch.from_numpy((metadata - mean) / std).to(device),
                    indices,
                    sample["ages"],
                )

            prepared = [prepare(sample) for sample in method_samples]

            def forward(value):
                patches, metadata, _, _ = value
                if patches.shape[0] == 0:
                    return np.empty(0, dtype=np.float32)
                with torch.inference_mode():
                    return model(patches, metadata)

            prepared_scores = []
            for value in prepared:
                output = forward(value)
                if isinstance(output, torch.Tensor):
                    output = output.detach().cpu().numpy()
                prepared_scores.append((value, np.asarray(output)))

            def rank(value_score):
                value, score_values = value_score
                _, _, indices, ages = value
                if method == "Age":
                    return select_top_k(ages, ratio)
                if method == "Pure CNN":
                    return select_top_k(score_values, ratio)
                score_map = {
                    int(index): float(score)
                    for index, score in zip(indices, score_values)
                }

                def score(candidate):
                    return np.asarray(
                        [score_map[int(index)] for index in candidate],
                        dtype=np.float32,
                    )

                scheduler = select_afvt if method == "AFVT" else select_magvr
                return scheduler(ages, ratio, score).mask

            def total(sample):
                if method == "Age":
                    return select_top_k(sample["ages"], ratio)
                value = prepare(sample)
                output = forward(value)
                if isinstance(output, torch.Tensor):
                    output = output.detach().cpu().numpy()
                return rank((value, np.asarray(output)))

            prep_stats = _measure_wall(
                prepare,
                method_samples,
                args.warmup,
                args.repeats,
                device,
            )
            model_stats = _measure_model(
                forward,
                prepared,
                max(200, args.warmup),
                args.repeats,
                device,
            )
            rank_stats = _measure_wall(
                rank,
                prepared_scores,
                args.warmup,
                args.repeats,
                device,
            )
            total_stats = _measure_wall(
                total,
                method_samples,
                args.warmup,
                args.repeats,
                device,
            )
            candidate_counts = np.asarray(
                [value[0].shape[0] for value in prepared],
                dtype=np.float64,
            )
            rows.append(
                {
                    "method": method,
                    "budget": ratio,
                    "params": 0 if method == "Age" else 5737,
                    "mean_cnn_blocks": float(candidate_counts.mean()),
                    "p95_cnn_blocks": float(
                        np.quantile(candidate_counts, 0.95)
                    ),
                    "frames_using_cnn_ratio": float(
                        (candidate_counts > 0).mean()
                    ),
                    "input_preparation_ms": prep_stats[0],
                    "model_forward_ms": model_stats[0],
                    "ranking_ms": rank_stats[0],
                    "total_selector_ms": total_stats[0],
                    "median_total_ms": total_stats[1],
                    "p95_total_ms": total_stats[2],
                    "p99_total_ms": total_stats[3],
                    "conditional_model_median_ms": model_stats[1],
                    "conditional_model_p95_ms": model_stats[2],
                }
            )

    output_root = PROJECT_ROOT / args.output_root
    _write_csv(output_root / "selector_overhead.csv", rows)
    (output_root / "config.yaml").write_text(
        f"warmup: {args.warmup}\nrepeats: {args.repeats}\n"
        f"device: {device}\nrepresentative_inputs: frozen_test_sequences\n"
        "patch_preparation: candidate_only\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(
        " ".join(sys.argv) + "\n", encoding="utf-8"
    )
    print(rows)


if __name__ == "__main__":
    main()
