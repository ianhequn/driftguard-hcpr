# DriftGuard Round 8 Source Recovery

日期：2026-08-25

## Result

```text
ROUND 8 SOURCE RECOVERY = PASS
ROUND 8 RESUME = AUTHORIZED
```

历史 `ROUND 8 = SOURCE CHECKPOINT BLOCKED` 保留在 `reports/round8_final_report.md` 与 `reports/round8_state.json`，本报告记录新的 recovery attempt。

## Source identity

- Source identifier: `nvidia/segformer-b1-finetuned-ade-512-512`
- Download endpoint: `https://hf-mirror.com`
- Resolved revision: `a775b4f5dcb2179e57006dca0e268281306412ef`
- Source task: ADE20K semantic segmentation
- Source classes: 150
- Processor: `SegformerImageProcessor`, 512x512

## Architecture and forward gate

Fresh-process loading passed. The source is `SegformerForSemanticSegmentation`, `model_type=segformer`, with B1 configuration:

```text
depths = [2, 2, 2, 2]
hidden_sizes = [64, 128, 320, 512]
attention_heads = [1, 2, 5, 8]
sr_ratios = [8, 4, 2, 1]
decoder_hidden_size = 256
```

A real CUDA forward on `1 x 3 x 512 x 512` produced finite logits with shape `[1, 150, 128, 128]`.

## Weight format and provenance

The official mirror file list contains `pytorch_model.bin`, not a native `model.safetensors`. The original file was retained and hashed. A deterministic `safetensors.torch.save_file` conversion was created only to match the project's loader contract; it is explicitly recorded as derived, not as the original source format.

| File | Bytes | SHA256 |
|---|---:|---|
| `config.json` | 6,885 | `8f1b7bcd2e1835ef0ef3a04a0d4dfe87ff9a6374253e7a3ca1751de812a420ff` |
| `preprocessor_config.json` | 271 | `8039d1d210abaa7117ad78e58cdfd6141a2ec72c03dae891b3cd76737e422c6c` |
| original `pytorch_model.bin` | 54,944,801 | `ac49bf62c8c0578dcf17ff048930b4f886ab0bc080d23f5591669afad5ddf7da` |
| derived `model.safetensors` | 54,889,568 | `4dbca19129b1373ca6bce75d68834a246d6b1e5e5202886183cd284d2f41abef` |

Full machine-readable provenance is in `results/R8R_E00_source_recovery/source_provenance.json` and `source_hashes.json`.
