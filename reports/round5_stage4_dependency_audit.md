# DriftGuard Round 5 Stage 4 Dependency Audit

## Installed Source

This audit uses the actually installed environment:

```text
PyTorch: 2.1.2+cu118
Transformers: 4.44.2
Model class: SegformerForSemanticSegmentation
Encoder class: SegformerEncoder
Stage 4 shape: 1x256x16x16 (256 tokens)
Stage 4 depth: 2 transformer blocks
Stage 4 heads: 8
Stage 4 sr_ratio: 1
```

The captured class source and forward signatures are stored in
`results/R5_E01_stage4_dependency/installed_source.txt` and
`installed_contract.json`.

## Actual Forward Graph

```text
fresh Stage 3 output: [B, 160, 32, 32]
  -> overlapping patch embedding
     Conv2d 3x3, stride 2, padding 1, 160 -> 256
     flatten to 256 tokens
     token-wise LayerNorm
  -> Transformer block 1
     token-wise LayerNorm
     Q projection for every query token
     K projection for every context token
     V projection for every context token
     global QK^T attention (sr_ratio = 1)
     softmax over all 256 keys
     attention-value matmul
     output projection
     residual add
     token-wise LayerNorm
     MixFFN Linear 256 -> 1024
     depthwise Conv2d 3x3 on the 16x16 token grid
     GELU
     Linear 1024 -> 256
     residual add
  -> Transformer block 2 (same dependency pattern)
  -> token-wise final LayerNorm
  -> Stage 4 output: [B, 256, 16, 16]
```

The decoder consumes all four encoder stages. It projects F1, F2, F3, and F4
to 256 channels, upsamples all of them to the F1 resolution, concatenates
them, fuses them, and produces logits. Round 5 must therefore retain fresh
F1-F3 and use only partial/cache semantics for F4.

## Dependency Matrix

| Operator | Spatial dependency | Global dependency | Cacheable? | Exact sparse possible? | Approx sparse possible? | Expected benefit |
|---|---|---|---|---|---|---|
| Stage 4 patch Conv 3x3/s2 | 3x3 Stage-3 neighborhood per output token | No | Output tokens | Yes, selected output positions with input halo | Yes | Moderate |
| Patch LayerNorm | One token | No | Normalized token | Yes | Yes | Small |
| Block LayerNorm 1 | One token | No | Normalized token | Yes | Yes | Small |
| Q projection | One query token | No | Q per layer | Yes | Yes | Moderate |
| K projection | One context token | All K are consumed by every query | K per layer | Only if current K for all required context is available | Cached stale K for non-refresh tokens | Moderate |
| V projection | One context token | All V are consumed by every query | V per layer | Only if current V for all required context is available | Cached stale V for non-refresh tokens | Moderate |
| Attention QK matmul | Selected query against 256 keys | Yes | Attention weights are not stable across current Q/K | Selected-query exact only with all current K | Selected query with mixed fresh/stale K | High |
| Softmax | All 256 keys for each query | Yes | No useful persistent cache | Yes per selected query with current logits | Yes with cached context | Moderate |
| Attention AV matmul | Selected query against 256 values | Yes | No useful output cache for changed Q/K/V | Selected-query exact only with all current V | Selected query with mixed fresh/stale V | High |
| Attention output Linear | One query token | No | Output token | Yes | Yes | Moderate |
| Residual adds | One token | No | Hidden token | Yes | Yes | Small |
| Block LayerNorm 2 | One token | No | Normalized token | Yes | Yes | Small |
| MixFFN dense1 | One token | No | 1024-D intermediate | Yes | Yes | Moderate |
| MixFFN depthwise 3x3 | One-cell token halo | No beyond halo | Dense1 intermediate | Yes with exact current halo | Yes with cached stale halo | Moderate |
| GELU / dense2 | One token | No | Output token | Yes | Yes | Moderate |
| Final LayerNorm | One token | No | Final token | Yes | Yes | Small |

## Attention Questions

### Q1: What attention is used?

Stage 4 uses global self-attention. Although the class is named
`SegformerEfficientSelfAttention`, the B0 Stage-4 configuration has
`sr_ratio=1`, so no spatial-reduction convolution is created or executed.

### Q2: Does a query subset still need all K/V?

Yes. A refreshed query subset can reduce Q-side work, but every refreshed
query attends over all 256 Stage-4 tokens and therefore needs `K_all` and
`V_all`.

### Q3: Can K/V be cached?

Yes, as historical context. Replacing non-refresh current K/V with cached K/V
is computationally useful but approximate. Exact current K/V require current
hidden states for all context tokens.

### Q4: Do refreshed tokens affect non-refresh outputs?

Yes. In a full current forward, changed refreshed K/V alter the attention
result of every query, including non-refresh queries. Holding non-refresh
outputs fixed is therefore an explicit approximation.

### Q5: Does exact current output require large global recomputation?

Yes. Block 2 needs current K/V for all tokens, which depend on exact block-1
outputs for all tokens. Therefore exact final outputs for even a query subset
require the complete patch embedding and complete first transformer block,
plus all-token K/V preparation in the second block.

## MixFFN Halo

Each transformer block contains one depthwise 3x3 convolution, so a refreshed
token's FFN output depends on a one-cell halo in that block's dense1
intermediate. Across two blocks, local receptive-field expansion alone would
reach two cells. Global attention is stronger: after the first attention,
every token can depend on every Stage-4 token.

For exact final selected tokens, the second block needs exact current
one-cell-halo attention outputs. For an approximate route, the halo can be
served from cached dense1 intermediates, but that must be reported as stale
cached context rather than exact current computation.

## Feasibility Decision

```text
EXACT PARTIAL:
COMPUTATIONALLY POINTLESS

APPROXIMATE PARTIAL:
FEASIBLE
```

An exact implementation is mathematically constructible, but its mandatory
lower bound includes:

```text
full Stage-4 patch embedding
+ full transformer block 1
+ all-token K/V preparation in transformer block 2
+ selected/halo query, attention, and FFN work in block 2
```

This is near-full Stage-4 work before PyTorch gather/scatter overhead, and the
complete model has only 20.43% Stage-4 runtime to recover. Route E is therefore
stopped as `EXACT PARTIAL NO-GO`.

One Route A is authorized: cached-context approximate partial with fresh
selected Q/K/V and hidden states, stale cached K/V for non-refresh tokens, and
cached MixFFN neighbor context. It must pass 100% equivalence, physical compute
instrumentation, latency, quality, and memory tests.
