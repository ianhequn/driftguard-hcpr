"""Profile end-to-end selector overhead for Round 2.5 deployable methods."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable

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
from driftguard.analysis.refresh_semantics import select_top_k  # noqa: E402
from driftguard.analysis.sequential_stale import decode_features  # noqa: E402
from driftguard.models.risk_predictor import TinyRiskCNN  # noqa: E402
from driftguard.models.segformer import (  # noqa: E402
    get_stage_features,
    load_segformer,
    prepare_inputs,
)
from driftguard.utils.common import choose_device  # noqa: E402
from driftguard.utils.video import iter_video_frames  # noqa: E402
from generate_round2_task_dataset import _patches  # noqa: E402
from train_round25_old_mlp import HistoricalTinyMLP  # noqa: E402


Stats = tuple[float, float, float, float]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stats(timings: list[float]) -> Stats:
    values = np.asarray(timings, dtype=np.float64)
    return (
        float(values.mean()),
        float(np.median(values)),
        float(np.quantile(values, 0.95)),
        float(np.quantile(values, 0.99)),
    )


def _measure_wall(
    fn: Callable,
    inputs: list,
    repeats: int,
    device: torch.device,
    warmup: int,
) -> Stats:
    for index in range(warmup):
        fn(inputs[index % len(inputs)])
    timings = []
    for index in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn(inputs[index % len(inputs)])
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return _stats(timings)


def _measure_cuda_or_wall(
    fn: Callable,
    inputs: list,
    repeats: int,
    device: torch.device,
    warmup: int,
) -> Stats:
    for index in range(warmup):
        fn(inputs[index % len(inputs)])
    timings = []
    for index in range(repeats):
        value = inputs[index % len(inputs)]
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
    return _stats(timings)


def _load_mlp(path: Path, device: torch.device):
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
    model = TinyRiskCNN(metadata_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    mean = np.asarray(checkpoint["metadata_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["metadata_std"], dtype=np.float32)
    std[std < 1e-8] = 1.0
    indices = list(checkpoint.get("metadata_indices", range(metadata_dim)))
    return model, mean, std, indices


def _sample_inputs(
    inventory: Path,
    model_name: str,
    device: torch.device,
    frames_per_video: int,
):
    with inventory.open(encoding="utf-8", newline="") as handle:
        records = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    processor, model = load_segformer(model_name, device=device)
    samples = []
    for record in records:
        frames = list(
            iter_video_frames(
                PROJECT_ROOT / record["path"],
                max_frames=frames_per_video,
            )
        )
        previous_frame, current_frame = frames[0][1], frames[1][1]
        with torch.inference_mode():
            previous_inputs = prepare_inputs(
                processor,
                Image.fromarray(previous_frame, mode="RGB"),
                device,
            )
            previous_outputs = model(
                pixel_values=previous_inputs["pixel_values"],
                output_hidden_states=True,
                return_dict=True,
            )
            previous_features = get_stage_features(previous_outputs, model=model)
            previous_logits = decode_features(model, previous_features).detach()
        samples.append(
            {
                "current": current_frame,
                "previous": previous_frame,
                "previous_logits": previous_logits,
                "age": np.arange(16, dtype=np.float32) % 5,
            }
        )
    return samples


def _all_metadata(sample: dict) -> np.ndarray:
    cheap = rgb_block_observables(sample["current"], sample["previous"], grid=4)
    entropy = entropy_block_observables(
        sample["previous_logits"],
        sample["current"].shape[0],
        sample["current"].shape[1],
        grid=4,
    )
    return np.column_stack(
        [
            sample["age"],
            entropy,
            np.full(16, cheap[0, 2], dtype=np.float32),
            cheap[:, 0],
            cheap[:, 3],
        ]
    ).astype(np.float32)


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
        default="results/R25_E03_selector_profiling",
    )
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--model-only-recheck", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 100 or args.repeats < 500:
        raise ValueError("Round 2.5 profiling requires warmup>=100 and repeats>=500")
    device = choose_device(args.device)
    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    old, old_mean, old_std = _load_mlp(PROJECT_ROOT / args.old_mlp, device)
    cnn_all, all_mean, all_std, all_indices = _load_cnn(
        PROJECT_ROOT / args.cnn_all,
        5,
        device,
    )
    cnn_pa, pa_mean, pa_std, pa_indices = _load_cnn(
        PROJECT_ROOT / args.cnn_patch_age,
        1,
        device,
    )
    if args.model_only_recheck:
        path = output_root / "selector_overhead.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        model_inputs = {
            "Old Tiny MLP": torch.zeros((16, 5), device=device),
            "CNN All": (
                torch.zeros((16, 3, 16, 16), device=device),
                torch.zeros((16, 5), device=device),
            ),
            "CNN Patch+Age": (
                torch.zeros((16, 3, 16, 16), device=device),
                torch.zeros((16, 1), device=device),
            ),
        }
        model_fns = {
            "Old Tiny MLP": lambda values: old(values),
            "CNN All": lambda values: cnn_all(*values),
            "CNN Patch+Age": lambda values: cnn_pa(*values),
        }
        recheck_warmup = max(200, args.warmup)
        for name, values in model_inputs.items():
            if device.type == "cuda":
                prime = torch.randn((2048, 2048), device=device)
                for _ in range(20):
                    prime = prime @ prime
                torch.cuda.synchronize()
                del prime

            def forward(value, fn=model_fns[name]):
                with torch.inference_mode():
                    return fn(value)

            stats = _measure_cuda_or_wall(
                forward,
                [values],
                args.repeats,
                device,
                recheck_warmup,
            )
            row = next(row for row in rows if row["method"] == name)
            row["model_forward_ms"] = stats[0]
            row["model_only_median_ms"] = stats[1]
            row["model_only_p95_ms"] = stats[2]
            row["model_only_p99_ms"] = stats[3]
        _write_csv(path, rows)
        (output_root / "model_only_recheck.txt").write_text(
            f"warmup: {recheck_warmup}\nrepeats: {args.repeats}\n"
            "inputs: synthetic tensors with production batch shapes\n"
            "reason: isolate CUDA forward from CPU-heavy input preparation\n",
            encoding="utf-8",
        )
        print(rows)
        return

    samples = _sample_inputs(
        PROJECT_ROOT / args.inventory,
        args.model,
        device,
        frames_per_video=2,
    )

    def prep_residual_mean(sample):
        return rgb_block_observables(
            sample["current"], sample["previous"], grid=4
        )[:, 0]

    def prep_residual_p90(sample):
        patches = _patches(sample["current"], sample["previous"], 4, 16)
        return np.quantile(patches.reshape(16, -1), 0.90, axis=1)

    def prep_age(sample):
        return sample["age"].copy()

    def prep_old(sample):
        metadata = _all_metadata(sample)[:, [3, 0, 1, 2, 4]]
        return torch.from_numpy((metadata - old_mean) / old_std).to(device)

    def prep_cnn_all(sample):
        patches = torch.from_numpy(
            _patches(sample["current"], sample["previous"], 4, 16)
        ).to(device)
        metadata = _all_metadata(sample)
        selected = torch.from_numpy(
            (metadata[:, all_indices] - all_mean) / all_std
        ).to(device)
        return patches, selected

    def prep_cnn_pa(sample):
        patches = torch.from_numpy(
            _patches(sample["current"], sample["previous"], 4, 16)
        ).to(device)
        metadata = sample["age"][:, None]
        selected = torch.from_numpy(
            (metadata[:, pa_indices] - pa_mean) / pa_std
        ).to(device)
        return patches, selected

    def no_forward(values):
        return values

    def old_forward(values):
        with torch.inference_mode():
            return old(values)

    def all_forward(values):
        with torch.inference_mode():
            return cnn_all(*values)

    def pa_forward(values):
        with torch.inference_mode():
            return cnn_pa(*values)

    specs = (
        ("Residual Mean", 0, prep_residual_mean, no_forward),
        ("Residual P90", 0, prep_residual_p90, no_forward),
        ("Age", 0, prep_age, no_forward),
        ("Old Tiny MLP", 737, prep_old, old_forward),
        ("CNN All", 5865, prep_cnn_all, all_forward),
        ("CNN Patch+Age", 5737, prep_cnn_pa, pa_forward),
    )
    rows = []
    for name, params, prepare, forward in specs:
        prepared = [prepare(sample) for sample in samples]

        def ranking(score):
            if isinstance(score, torch.Tensor):
                score = score.detach().cpu().numpy()
            return select_top_k(np.asarray(score), 0.375)

        def total(sample):
            values = prepare(sample)
            score = forward(values)
            return ranking(score)

        prep_stats = _measure_wall(
            prepare, samples, args.repeats, device, args.warmup
        )
        if name in {"Residual Mean", "Residual P90", "Age"}:
            forward_stats = (0.0, 0.0, 0.0, 0.0)
            scores = prepared
        else:
            forward_stats = _measure_cuda_or_wall(
                forward, prepared, args.repeats, device, args.warmup
            )
            scores = [forward(value).detach() for value in prepared]
        ranking_stats = _measure_wall(
            ranking, scores, args.repeats, device, args.warmup
        )
        total_stats = _measure_wall(
            total, samples, args.repeats, device, args.warmup
        )
        rows.append(
            {
                "method": name,
                "params": params,
                "input_preparation_ms": prep_stats[0],
                "model_forward_ms": forward_stats[0],
                "ranking_ms": ranking_stats[0],
                "total_selector_ms": total_stats[0],
                "median_total_ms": total_stats[1],
                "p95_total_ms": total_stats[2],
                "p99_total_ms": total_stats[3],
                "model_only_median_ms": forward_stats[1],
                "model_only_p95_ms": forward_stats[2],
                "model_only_p99_ms": forward_stats[3],
            }
        )

    _write_csv(output_root / "selector_overhead.csv", rows)
    (output_root / "config.yaml").write_text(
        f"repeats: {args.repeats}\nwarmup: {args.warmup}\ndevice: {device}\n"
        "timing: cuda_event_for_model_forward_wall_clock_for_cpu_and_total\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    print(rows)


if __name__ == "__main__":
    main()
