# DriftGuard Round 7.6 Restart Frozen Model Record

## Frozen artifact

- Directory: `/root/autodl-tmp/models/segformer-b0-vspw124`
- Architecture: SegFormer-B0
- `num_labels`: 124
- Selected step: 40000
- Selected validation mIoU: 0.3012895793848641
- Fresh-process validation mIoU: 0.30129196806407454
- Absolute reload delta: 0.00000238867921042996
- Checkpoint SHA256:
  `e8208b7075e10f6418bcacaddfcc7a52114c04380f9ff88079d35296ad3eefdd`
- Frozen `model.safetensors` SHA256:
  `8b673cb6bc173f3e986da8446c2c3debacc77779109b4916b230af36f197f17b`

## Contract and alignment

The frozen artifact records the canonical contract: official VSPW
normalization, semantic IDs 0..123, and ignore ID 255. Raw 253 follows
`253 -> 252 -> 255` and is excluded. The model output cannot contain 255;
255 exists only in targets and evaluation masks.

Class alignment is verified as 124 / 124. The model forward emits 124 logits.

## Architecture audit

| Stage | Feature shape |
|---:|---|
| 1 | [1, 32, 128, 128] |
| 2 | [1, 64, 64, 64] |
| 3 | [1, 160, 32, 32] |
| 4 | [1, 256, 16, 16] |

The decoder consumes all four encoder stages and its classifier emits 124
channels. Structured whole-Stage-4 reuse compatibility is `DIRECT`. No
temporal reuse was executed in this round.

## QA

Twenty validation frames were exported as RGB, canonical GT, and prediction
images under the cloud artifact directory
`results/R76R_E06_architecture_audit/qa/`, with a manifest retained locally.
The frozen validation prediction histogram contains four never-predicted
classes (43, 47, 95, 113) and three very rare prediction classes (4, 72, 73).
This is recorded as a long-tail coverage risk, not a global output collapse;
all 124 classes occur in the canonical validation support and the model emits
124 logits.
