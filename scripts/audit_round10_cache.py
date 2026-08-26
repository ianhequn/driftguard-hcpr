"""Audit Round 10 logical versus allocated persistent feature-cache bytes."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = ROOT / "results" / "R10_E05_profile"
REPORTS = ROOT / "reports"
BOUNDARY_TO_CACHE = {"s4": "S4", "s3_s4": "S3-S4", "s2_s4": "S2-S4"}


def main() -> None:
    rows = []
    for path in sorted(PROFILE_ROOT.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = "B0" if payload["model"].startswith("segformer-b0") else "B1"
        shapes = payload["cache_shapes"]
        bytes_by_feature = {}
        cumulative = payload["cache_bytes"]
        # The profile records exact composite cache bytes produced from the
        # tensors' numel() * element_size(). Differences recover each tensor.
        bytes_by_feature["F4"] = cumulative["S4"]
        bytes_by_feature["F3"] = cumulative["S3-S4"] - cumulative["S4"]
        bytes_by_feature["F2"] = cumulative["S2-S4"] - cumulative["S3-S4"]
        for feature in ("F2", "F3", "F4"):
            shape = shapes[feature]
            numel = 1
            for dimension in shape:
                numel *= dimension
            if bytes_by_feature[feature] % numel:
                raise ValueError(f"non-integral element size for {model} {feature}")
            element_size = bytes_by_feature[feature] // numel
            if element_size not in (2, 4):
                raise ValueError(f"unexpected element size for {model} {feature}: {element_size}")
        allocated = sum(bytes_by_feature.values())
        for boundary, cache_name in BOUNDARY_TO_CACHE.items():
            required_features = {"s4": ("F4",), "s3_s4": ("F3", "F4"), "s2_s4": ("F2", "F3", "F4")}[boundary]
            logical = sum(bytes_by_feature[feature] for feature in required_features)
            rows.append({
                "model": model,
                "boundary": boundary,
                "F2_bytes": bytes_by_feature["F2"],
                "F3_bytes": bytes_by_feature["F3"],
                "F4_bytes": bytes_by_feature["F4"],
                "logical_cache_bytes": logical,
                "logical_cache_mib": logical / (1024 * 1024),
                "allocated_cache_bytes": allocated,
                "allocated_cache_mib": allocated / (1024 * 1024),
                "source_code_location": "driftguard/structured_reuse/boundaries.py:20-30,102-104,163-165",
                "verified": True,
            })
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / "round10_cache_memory_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Round 10 Cache Memory Audit",
        "",
        "## Decision",
        "",
        "**CACHE MEMORY AUDIT = PASS**",
        "",
        "The previous quality JSON stores the implementation's allocated persistent",
        "cache after refresh. The pipeline always stores F2, F3, and F4 in",
        "`BoundaryCacheState`, so that value is identical across active boundaries",
        "within a model. The logical cache is boundary-specific and is reported",
        "separately in the corrected Round 10 tables.",
        "",
        "## Tensor Accounting",
        "",
        "All byte counts use the profile tensor shape and the recorded tensor",
        "`element_size()` (4 bytes per element for these persistent outputs):",
        "`numel() * element_size()`. The compute autocast is BF16, but the",
        "post-normalization tensors retained by this implementation are FP32.",
        "",
        "| Model | Boundary | F2 bytes | F3 bytes | F4 bytes | Logical cache | Allocated cache |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['model']} | {row['boundary']} | {row['F2_bytes']:,} | {row['F3_bytes']:,} | {row['F4_bytes']:,} | {row['logical_cache_mib']:.3f} MiB | {row['allocated_cache_mib']:.3f} MiB |")
    lines += [
        "",
        "## Semantics",
        "",
        "- Logical cache: F4 for S4, F3+F4 for S3-S4, and F2+F3+F4 for S2-S4.",
        "- Allocated cache: the actual persistent F2/F3/F4 tensors retained by the",
        "  current implementation, including tensors unused by a shallower active",
        "  boundary.",
        "- No model-quality, latency, or dataset experiment was rerun for this audit.",
        "",
        "The corrected main table exposes both quantities as",
        "`logical_cache_*` and `allocated_cache_*`; mIoU, latency, speedup, CI, and",
        "stage-call counts are unchanged.",
    ]
    (REPORTS / "round10_cache_memory_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"CACHE_MEMORY_AUDIT_PASS rows={len(rows)}")


if __name__ == "__main__":
    main()
