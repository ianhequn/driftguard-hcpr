# DriftGuard Round 7.6 Initialization Audit

## Source

```text
checkpoint: /root/autodl-tmp/models/segformer-b0
source task: ADE20K semantic segmentation
source classes: 150
model SHA256: 6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9
architecture: SegFormer-B0
encoder channels: 32 / 64 / 160 / 256
encoder depths: 2 / 2 / 2 / 2
decoder hidden size: 256
```

The source checkpoint is initialization only. Its ADE20K classifier and class
metadata are not used as the final VSPW output taxonomy.

## Explicit Load

The VSPW model is constructed with `num_labels=124`. The source safetensors
state is loaded after explicitly removing:

```text
decode_head.classifier.weight
decode_head.classifier.bias
```

Measured initialization result:

```text
loaded parameter tensors: 206
missing keys: classifier weight and bias only
unexpected keys: 0
reinitialized parameters: classifier weight and bias
trainable parameters: 3,746,012
total parameters: 3,746,012
encoder frozen: NO
```

One real VSPW frame produced finite logits with shape `[1,124,128,128]`.
Stage outputs at 512x512 were `[32,128,128]`, `[64,64,64]`,
`[160,32,32]`, and `[256,16,16]`.
