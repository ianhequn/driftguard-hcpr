# Round 7.5 Readiness Checklist

```text
[ ] official VSPW package installed
[ ] package integrity checked
[ ] official split known
[ ] real mask semantics verified
[ ] 124 classes verified from installed masks
[ ] ignore index 255 verified from installed masks
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
DATASET ACCESS BLOCKED
NO VERIFIED VSPW-COMPATIBLE CHECKPOINT
```
