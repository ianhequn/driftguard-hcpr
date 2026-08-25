# Round 7 Restart: Equivalence and Contract

## Frozen artifact

- Model: `/root/autodl-tmp/models/segformer-b0-vspw124`
- SHA256: `8b673cb6bc173f3e986da8446c2c3debacc77779109b4916b230af36f197f17b`
- `num_labels`: 124
- Input shape: `1 x 3 x 512 x 512`

## Validation

`results/R7R_E00_preflight/equivalence.json` records 20 validation frames. Full native SegFormer and the whole-stage reuse pipeline with `refresh=True` produced:

| Metric | Result |
|---|---:|
| Stage shapes | `[1,32,128,128]`, `[1,64,64,64]`, `[1,160,32,32]`, `[1,256,16,16]` |
| Max absolute error | 0 |
| Mean absolute error | 0 |
| Relative L2 | 0 |
| Prediction disagreement | 0 |
| Allclose | true |
| Stage4 calls | 20 |
| Decoder frames | 20 |

This is a refresh-equivalence gate, not a claim that cached reuse is mathematically current-frame exact. Reuse intentionally returns the cached final F4.

## Runtime contract

Stage1-3 remain fresh for every frame. Stage4 is dense only on refresh frames. The decoder runs for every frame. The first frame of each video refreshes, and the cache is reset between videos. The formal benchmark confirms FI-2 executed 12,289 Stage4 calls for 24,392 frames.

## Label contract

Official raw VSPW labels use the frozen normalization `0 -> 255; subtract 1; 254 -> 255`, followed by mapping normalized IDs outside `[0,123]` to ignore `255`. Raw `253` therefore becomes normalized `252` and is excluded as ignore. No pseudo-ground-truth was introduced.
