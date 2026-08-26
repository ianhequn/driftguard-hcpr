"""Non-formal throughput diagnostic for Round 10 batch execution."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to("cuda").eval()
    pixels = torch.randn(args.batch, 3, 512, 512, device="cuda", dtype=torch.float32)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(10):
            model(pixel_values=pixels)
        torch.cuda.synchronize()
        start = time.perf_counter()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(args.iterations):
            model(pixel_values=pixels)
        end.record()
        end.synchronize()
        cuda_ms = begin.elapsed_time(end) / args.iterations
    print({"batch": args.batch, "iterations": args.iterations, "cuda_ms": cuda_ms,
           "wall_ms": (time.perf_counter() - start) * 1000 / args.iterations})


if __name__ == "__main__":
    main()
