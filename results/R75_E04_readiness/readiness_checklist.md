# Round 7.5 Readiness Checklist

```text
[x] official VSPW package installed
[x] package integrity checked
[x] official split known
[x] real mask semantics verified
[x] 124 classes verified from installed taxonomy and masks
[x] ignore index 255 verified from installed masks
[ ] compatible checkpoint found
[ ] checkpoint source recorded
[ ] checkpoint SHA256 recorded
[ ] checkpoint trained on VSPW
[ ] exact VSPW class order verified
[ ] checkpoint loads
[ ] forward smoke passes
[ ] output has 124 classes
[ ] data/model preprocessing compatible
[ ] Round 7 evaluator can accept pair
[x] no temporal reuse benchmark has yet been run
```

Configuration-only reference:

```text
ziplab/MPVSS:
124 classes and ignore 255 verified from config,
but checkpoint unavailable and architecture requires major adaptation.
```

```text
BENCHMARK READINESS = NOT READY
```

Blocking conditions:

```text
NO VERIFIED VSPW-COMPATIBLE CHECKPOINT
```

Dataset notes:

```text
official train/val/test videos: 2806 / 343 / 387
official test GT: not included
GT benchmark split: official validation
official damaged-frame repair: applied and verified
```
