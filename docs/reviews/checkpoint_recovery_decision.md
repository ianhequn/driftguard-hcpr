# Checkpoint recovery decision

## Decision

**EXACT HISTORICAL CHECKPOINTS NOT RECOVERED**

The required full-file SHA256 values are:

- B0: `8b673cb6bc173f3e986da8446c2c3debacc77779109b4916b230af36f197f17b`
- B1: `892a6589874d762c6b9c911e559e062fccbb226b33309aa6c8317dbec091fcfb`

The read-only inventory in `reports/post_mmm/checkpoint_inventory.json` found no file with either hash. No candidate was copied into `recovered_checkpoints/`.

## Provenance answers

| Question | Evidence-backed answer |
|---|---|
| B0 base checkpoint | `/root/autodl-tmp/models/segformer-b0`, ADE-derived initialization SHA256 `6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9`; classifier weight/bias were reinitialized. |
| B1 base checkpoint | `/root/autodl-tmp/models/segformer-b1-ade512-source`, converted source SHA256 `4dbca19129b1373ca6bce75d68834a246d6b1e5e5202886183cd284d2f41abef`; classifier weight/bias were reinitialized. |
| Data/splits/mapping | VSPW validation: 343 videos, 24,392 frames, 124 classes; official normalization followed by valid IDs `0..123`, otherwise ignore `255`. Raw `253` becomes normalized `252` and then `255`. |
| Training hyperparameters | Both historical B0/B1 records: 40,000 steps, batch 8, accumulation 1, AdamW, LR `6e-5`, weight decay `0.01`, linear warmup/decay, warmup ratio `0.02` (800 steps), BF16, seed 42, validation/save every 5,000 for B0 and validation every 10,000 for B1. `cudnn.benchmark=true`, deterministic=false. |
| Selection | Step 40,000 was selected for both models. B0 validation mIoU `0.3012895793848641`; B1 validation mIoU `0.37159828238560033`. |
| Original absolute path | Frozen directories were `/root/autodl-tmp/models/segformer-b0-vspw124` and `/root/autodl-tmp/models/segformer-b1-vspw124`. The files are not in this workstation. |
| Execution host | Records identify a cloud Linux/AutoDL environment, not a local Windows path. The old `D:\DriftGuard-r11a` and `D:\DriftGuard-r11b` worktrees are prunable metadata entries only. |
| Transfer history | The inspected local R11 archives contain results/tools/finalizer material but no target model member. `D:\DriftGuard_cloud_upload.tar.gz` is truncated; the clean upload archive has no checkpoint payload. No local scp/rsync record contains the frozen files. |
| Earliest/latest hash occurrence | In committed project evidence, B0 first appears in the 2026-08-25 Round 7.6 restart commit and latest in the 2026-08-27 R11B freeze commit. B1 first appears in the 2026-08-26 model-scale commit and latest in the same 2026-08-27 R11B freeze commit. These are metadata occurrences, not weight files. |
| Historical loading command | `SegformerForSemanticSegmentation.from_pretrained(<frozen-dir>, local_files_only=True)` was used by the Round 10/11 scripts; the frozen directory names above are the only historical absolute loading paths recorded. |
| Deterministic reproduction | Code and main hyperparameters are sufficient to start a new training run, but not to prove bitwise reproduction: the exact cloud commit, dataset byte manifest, dataloader/worker ordering, complete environment lock, and optimizer checkpoint are not available. A new run is a new lineage. |

## Candidate findings

The inventory hashed ordinary candidates and listed checkpoint-like archive members without extraction. Relevant non-exact findings:

- local ADE B0 base: `6ae39add...fee23f9`, `15,036,944` bytes, classified `BASE_MODEL_ONLY`;
- `D:\segformer_b0_hf_cache.tar.gz`: only contains the same ADE B0 cache member;
- BGE/whisper/SEE/old estimator files and unrelated `.pth/.bin` files have non-target hashes;
- R11A/R11B result archives list no `model.safetensors`, `pytorch_model.bin`, or VSPW frozen model member;
- WSL `ext4.vhdx` files were recorded but not mounted, copied, or modified;
- Git unreachable commits contain source/results metadata but no weight blob path matching the target model.

The ADE base model must not be substituted for either frozen VSPW-124 model. It has a different task head/label space and does not support the paper's historical quality numbers.

## Reproduction boundary

The new configs in `configs/post_mmm/` define a separate `POSTMMM-R1` lineage. They preserve evidence-backed values, mark unresolved fields as `TODO_EVIDENCE_REQUIRED`, and do not claim that retraining will reproduce either historical SHA. The first GPU run must write a new checkpoint SHA, git commit, environment fingerprint, dataset manifest hash, and protocol ID before any Kill Test A result is accepted.

No paper claim, historical result, or old checkpoint record was strengthened or rewritten.
