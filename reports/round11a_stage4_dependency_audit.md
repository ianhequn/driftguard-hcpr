# Round 11A Stage 4 Dependency Audit (segformer-b0-vspw124)

## Installed graph

Transformers model class: `SegformerForSemanticSegmentation`.
Encoder class: `SegformerEncoder`; Stage 4 contains `2` block(s).
Stage 4 attention uses `no spatial-reduction convolution (sr_ratio=1)`.

```text
Stage 3 output [B,C3,H3,W3]
  -> Stage 4 patch embedding (Conv2d + LayerNorm + flatten/transpose)
  -> 2 x Transformer block:
       LayerNorm -> Q/K/V projections -> attention score matmul -> softmax
       -> attention output projection -> residual
       -> LayerNorm -> MixFFN dense1 -> DWConv/activation -> dense2 -> residual
  -> Stage 4 final LayerNorm
  -> Stage 4 output F4 [B,C4,H4,W4]
  -> decoder (with F1,F2,F3,F4)
```

## Actual modules

- Patch embedding: `SegformerOverlapPatchEmbeddings`; see `stage4.json` for full repr.
- Attention modules per block: `['query', 'key', 'value', 'dropout']`.
- MixFFN modules per block: `['dense1', 'dwconv', 'intermediate_act_fn', 'dense2', 'dropout']`.

## Dependency matrix

| Operator | Spatial dependency | Global dependency | Cacheable? | Exact sparse possible? | Approx sparse possible? | Expected benefit |
|---|---|---|---|---|---|---|
| Stage 4 patch embedding | local convolutional neighborhood | no | only as whole-stage output | no for changed F3 | no useful token subset without boundary handling | dense stage skip only |
| LayerNorm | token-local | no | stale intermediate changes with F3 | yes only if all upstream tokens unchanged | yes, but stale | low |
| Q projection | token-local | no | no for exact current F3 | yes in isolation | yes | low |
| K/V projection | token-local when sr=1 | no | stale K/V gives approximate context | no for current changed tokens | yes | limited by global attention |
| Attention score matmul | all query/key tokens | yes | cached context is approximate | no, exact refreshed query still needs all current K/V | yes | no exact local saving |
| Softmax | all keys for each query | yes | no exact cache under changed K/V | no | yes | no exact local saving |
| Attention output projection | token-local after context | inherited global input | only with stale context | no end-to-end | yes | low alone |
| MixFFN Linear/activation | token-local | no | stale hidden is approximate | potentially, if input token is current | yes | moderate in isolation |
| Depthwise Conv | 1-cell spatial halo | no beyond halo | stale neighbors approximate | only with exact halo recomputation | yes | bounded by halo |
| Residual additions | token-local | inherited from branch | no if branch stale | not a source of saving | yes | none alone |

## Attention answers

1. Stage 4 attention classification: **global attention with `no spatial-reduction convolution (sr_ratio=1)`**; `sr_ratio=1` means it is not spatial-reduced in this checkpoint.
2. Updating only a query subset still requires K/V for all tokens because the attention score and softmax run over the full key sequence.
3. Current K/V can be cached only as stale context; it is not an exact current K/V cache after F3 changes.
4. Non-refresh token outputs can change through global attention and cannot be treated as independent exact values.
5. Exact current refreshed features therefore require essentially the full Stage 4 attention context and sequential block computation.

## MixFFN boundary

The depthwise convolution is a 3x3 operator in the installed MixFFN. A changed cell requires a one-cell halo for an exact local convolution at that block; with multiple blocks the halo expands per block, while global attention already dominates the exact dependency.

## Feasibility decision

- **EXACT PARTIAL: COMPUTATIONALLY POINTLESS.** Exact current Stage-4 values depend on global K/V context and sequential block state, so a token subset does not remove enough work to be meaningful.
- **APPROXIMATE PARTIAL: FEASIBLE in principle, but outside Round 11A.** Cached hidden/K/V context changes semantics and must be called approximate; Round 11A HCPR evaluates whole-stage dense refresh/reuse only.
- **EXACT PARTIAL NO-GO.** The validated route is whole-stage dense refresh or whole-stage cache reuse.
