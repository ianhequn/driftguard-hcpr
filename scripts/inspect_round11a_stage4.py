"""Inspect the installed SegFormer Stage-4 graph and write an audit report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def module_names(module):
    return list(module._modules.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).eval()
    encoder = model.segformer.encoder
    stage = 3
    blocks = list(encoder.block[stage])
    attention = [block.attention.self for block in blocks]
    mlps = [block.mlp for block in blocks]
    payload = {
        "model": args.model_dir.name,
        "encoder_class": type(encoder).__name__,
        "stage4_index": 4,
        "stage4_block_count": len(blocks),
        "stage4_spatial_reduction_ratio": int(encoder.config.sr_ratios[stage]),
        "patch_embedding": repr(encoder.patch_embeddings[stage]),
        "layer_norm": repr(encoder.layer_norm[stage]),
        "blocks": [{"class": type(block).__name__, "modules": module_names(block), "repr": repr(block)} for block in blocks],
        "attention": [{"class": type(value).__name__, "modules": module_names(value), "repr": repr(value)} for value in attention],
        "mlp": [{"class": type(value).__name__, "modules": module_names(value), "repr": repr(value)} for value in mlps],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    sr = payload["stage4_spatial_reduction_ratio"]
    reduction_text = "no spatial-reduction convolution (sr_ratio=1)" if sr == 1 else f"spatial reduction ratio {sr}"
    lines = [
        f"# Round 11A Stage 4 Dependency Audit ({args.model_dir.name})", "",
        "## Installed graph", "",
        f"Transformers model class: `{type(model).__name__}`.",
        f"Encoder class: `{type(encoder).__name__}`; Stage 4 contains `{len(blocks)}` block(s).",
        f"Stage 4 attention uses `{reduction_text}`.", "",
        "```text",
        "Stage 3 output [B,C3,H3,W3]",
        "  -> Stage 4 patch embedding (Conv2d + LayerNorm + flatten/transpose)",
        f"  -> {len(blocks)} x Transformer block:",
        "       LayerNorm -> Q/K/V projections -> attention score matmul -> softmax",
        "       -> attention output projection -> residual",
        "       -> LayerNorm -> MixFFN dense1 -> DWConv/activation -> dense2 -> residual",
        "  -> Stage 4 final LayerNorm",
        "  -> Stage 4 output F4 [B,C4,H4,W4]",
        "  -> decoder (with F1,F2,F3,F4)", "```", "",
        "## Actual modules", "",
        f"- Patch embedding: `{type(encoder.patch_embeddings[stage]).__name__}`; see `stage4.json` for full repr.",
        f"- Attention modules per block: `{module_names(attention[0])}`.",
        f"- MixFFN modules per block: `{module_names(mlps[0])}`.", "",
        "## Dependency matrix", "",
        "| Operator | Spatial dependency | Global dependency | Cacheable? | Exact sparse possible? | Approx sparse possible? | Expected benefit |",
        "|---|---|---|---|---|---|---|",
        "| Stage 4 patch embedding | local convolutional neighborhood | no | only as whole-stage output | no for changed F3 | no useful token subset without boundary handling | dense stage skip only |",
        "| LayerNorm | token-local | no | stale intermediate changes with F3 | yes only if all upstream tokens unchanged | yes, but stale | low |",
        "| Q projection | token-local | no | no for exact current F3 | yes in isolation | yes | low |",
        "| K/V projection | token-local when sr=1 | no | stale K/V gives approximate context | no for current changed tokens | yes | limited by global attention |",
        "| Attention score matmul | all query/key tokens | yes | cached context is approximate | no, exact refreshed query still needs all current K/V | yes | no exact local saving |",
        "| Softmax | all keys for each query | yes | no exact cache under changed K/V | no | yes | no exact local saving |",
        "| Attention output projection | token-local after context | inherited global input | only with stale context | no end-to-end | yes | low alone |",
        "| MixFFN Linear/activation | token-local | no | stale hidden is approximate | potentially, if input token is current | yes | moderate in isolation |",
        "| Depthwise Conv | 1-cell spatial halo | no beyond halo | stale neighbors approximate | only with exact halo recomputation | yes | bounded by halo |",
        "| Residual additions | token-local | inherited from branch | no if branch stale | not a source of saving | yes | none alone |", "",
        "## Attention answers", "",
        f"1. Stage 4 attention classification: **global attention with `{reduction_text}`**; `sr_ratio=1` means it is not spatial-reduced in this checkpoint.",
        "2. Updating only a query subset still requires K/V for all tokens because the attention score and softmax run over the full key sequence.",
        "3. Current K/V can be cached only as stale context; it is not an exact current K/V cache after F3 changes.",
        "4. Non-refresh token outputs can change through global attention and cannot be treated as independent exact values.",
        "5. Exact current refreshed features therefore require essentially the full Stage 4 attention context and sequential block computation.", "",
        "## MixFFN boundary", "",
        "The depthwise convolution is a 3x3 operator in the installed MixFFN. A changed cell requires a one-cell halo for an exact local convolution at that block; with multiple blocks the halo expands per block, while global attention already dominates the exact dependency.", "",
        "## Feasibility decision", "",
        "- **EXACT PARTIAL: COMPUTATIONALLY POINTLESS.** Exact current Stage-4 values depend on global K/V context and sequential block state, so a token subset does not remove enough work to be meaningful.",
        "- **APPROXIMATE PARTIAL: FEASIBLE in principle, but outside Round 11A.** Cached hidden/K/V context changes semantics and must be called approximate; Round 11A HCPR evaluates whole-stage dense refresh/reuse only.",
        "- **EXACT PARTIAL NO-GO.** The validated route is whole-stage dense refresh or whole-stage cache reuse.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model_dir.name, "blocks": len(blocks), "sr_ratio": sr, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
