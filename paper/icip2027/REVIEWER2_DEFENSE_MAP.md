# Reviewer #2 Defense Map (Internal)

| Reviewer attack | Evidence in manuscript | Evidence in Round 10 artifacts | Best bounded response | Required revision | Residual risk |
| --- | --- | --- | --- | --- | --- |
| Temporal reuse and stage clocking are established. | Introduction and Related Work acknowledge DFF, Clockwork, Accel; Contributions frame a characterization. | reports/round10_reuse_boundary_claims.md | Agree that the mechanism is established; the contribution is isolating boundary x interval behavior under one measured hierarchy. | Complete. | Novelty remains borderline. |
| TFC already studies high-level reuse. | TFC positioning explicitly says S3--S4 is vanilla and omits TFC alignment/fusion/correction. | paper reference audit and official AAAI citation; Round 10 boundary definitions. | This is not a TFC reproduction or replacement; it is a controlled baseline family used to compare boundaries. | Complete. | Reviewer may still demand algorithmic novelty. |
| Only VSPW is used. | Dataset paragraph calls results validation-system characterization and Limitations narrows scope. | 343 videos, 24,392 pairs, 124 classes in round10 state and quality CSV. | Do not generalize beyond the declared VSPW validation protocol. | Complete. | External validity is intentionally limited. |
| One RTX 4090 D is insufficient for hardware claims. | Timing protocol and Conclusion name the GPU; Limitations reject portability. | profile table, five-repetition paired latency files, Round 10 state. | Hardware-aware means measuring GPU execution rather than FLOPs only, not portability across devices. | Complete. | “Hardware-aware” can be replaced by “GPU-measured” during human review if desired. |
| B0/B1 do not establish a model-scale law. | Paper says ordering is reproduced under two tested scales, not a scale-independent law. | B0/B1 Pareto audit and profile table. | Present this as replication across two scales, not universal scale behavior. | Complete. | Broader generalization remains unsupported. |
| Earlier boundary is obviously faster and less accurate. | Results explicitly enumerate dominated settings and the five-point frontier. | round10_pareto_audit.md and frontier CSV. | The nontrivial evidence is cross-setting dominance: earlier boundary plus shorter interval can beat later boundary plus longer interval in both criteria. | Complete. | Significance depends on reviewer valuing controlled characterization. |
| Fixed intervals are arbitrary. | FI-2/FI-4/FI-8 are labelled fixed study points. | Full 18-setting matrix. | The intervals are an experimental grid, not an adaptive scheduler or optimal policy. | Complete. | No continuous interval analysis. |
| Excluding decoding/H2D weakens practical latency. | CUDA-resident input-to-logits boundary and exclusions are explicit. | Round 10 latency protocol and state. | The claim is accelerator-path latency, not camera-to-display latency. | Complete. | End-to-end product latency remains unknown. |
| Pareto audit adds no insight. | Full matrix plus boundary-first empirical ordering. | independent audit recomputes dominance separately for B0/B1. | The audit exposes dominated policies and reproduces a common ordering across scales. | Complete. | It remains derived analysis rather than a new algorithm. |
| Five repeats are weak statistics. | Paper calls intervals bootstrap intervals over five paired repetitions. | latency artifacts contain five repetitions and paired Full baseline. | Treat intervals as uncertainty summaries for this repeated protocol, not population confidence guarantees. | Complete. | More repetitions could strengthen inference but are not required for the bounded claim. |
| Validation-selected checkpoints contaminate quality. | Explicit validation-system wording; no held-out test claim. | frozen checkpoint hashes and VSPW validation state. | Agree and narrow the claim; the paper is not presenting unbiased test generalization. | Complete. | Dataset selection bias remains. |
| S3--S4 is not a faithful prior-method reproduction. | Explicit vanilla baseline wording and no TFC modules. | boundary equations and code semantics. | It is a controlled reference condition, not a reproduction claim. | Complete. | Related-work comparison is conceptual rather than apples-to-apples. |
| Characterization without a new model is not publishable. | Contributions and Conclusion focus on empirical decision surface. | physical stage calls, cache audit, full matrix, paired latency. | The paper offers evidence for a systems-relevant question that is often obscured by compute accounting; judge it as measurement-driven ICIP work. | Complete. | Venue novelty threshold is the main unresolved risk. |

## Safe phrases

- “Under the tested VSPW validation protocol and RTX 4090 D...”
- “The measured frontier suggests...”
- “vanilla high-level feature-reuse baseline”
- “bootstrap interval over five repeated paired measurements”
- “validation-system characterization”
- “runtime opportunity, not a portability guarantee”

## Phrases to avoid

- “first”
- “novel architecture”
- “optimal boundary” without the tested-scope qualifier
- “real-time” for the proposed system
- “generalizes”
- “state of the art”
- “hardware-independent”
- “TFC reproduction”
- “model speedup” when only a component was measured
