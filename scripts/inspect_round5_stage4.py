"""Record the installed Hugging Face SegFormer Stage-4 implementation."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.models.segformer import load_segformer  # noqa: E402


CLASS_NAMES = (
    "SegformerOverlapPatchEmbeddings",
    "SegformerDWConv",
    "SegformerEfficientSelfAttention",
    "SegformerSelfOutput",
    "SegformerAttention",
    "SegformerMixFFN",
    "SegformerLayer",
    "SegformerEncoder",
    "SegformerDecodeHead",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/root/autodl-tmp/models/segformer-b0",
    )
    parser.add_argument(
        "--output-root",
        default="results/R5_E01_stage4_dependency",
    )
    return parser.parse_args()


def main() -> None:
    import torch
    import transformers
    from transformers.models.segformer import modeling_segformer

    args = parse_args()
    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    _, model = load_segformer(
        args.model,
        device=torch.device("cpu"),
        local_files_only=True,
    )
    encoder = model.segformer.encoder
    stage_index = 3
    config = model.config

    contract = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_path": args.model,
        "model_class": type(model).__name__,
        "encoder_class": type(encoder).__name__,
        "hidden_sizes": list(config.hidden_sizes),
        "depths": list(config.depths),
        "num_attention_heads": list(config.num_attention_heads),
        "sr_ratios": list(config.sr_ratios),
        "patch_sizes": list(config.patch_sizes),
        "strides": list(config.strides),
        "reshape_last_stage": bool(config.reshape_last_stage),
        "decoder_hidden_size": int(config.decoder_hidden_size),
        "stage4_patch_embedding": repr(encoder.patch_embeddings[stage_index]),
        "stage4_blocks": [
            repr(block) for block in encoder.block[stage_index]
        ],
        "stage4_layer_norm": repr(encoder.layer_norm[stage_index]),
        "decode_head": repr(model.decode_head),
        "forward_signatures": {},
    }

    source_sections = []
    for name in CLASS_NAMES:
        value = getattr(modeling_segformer, name, None)
        if value is None:
            source_sections.append(f"## {name}\n\nNOT PRESENT\n")
            continue
        contract["forward_signatures"][name] = str(
            inspect.signature(value.forward)
        )
        source_sections.append(
            f"## {name}\n\n"
            f"forward{inspect.signature(value.forward)}\n\n"
            f"{inspect.getsource(value)}\n"
        )

    (output_root / "installed_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "installed_source.txt").write_text(
        "\n".join(source_sections),
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(contract, indent=2))
    print(f"output_root={output_root.resolve()}")


if __name__ == "__main__":
    main()
