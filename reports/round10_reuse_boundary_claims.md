# Round 10 Claim Boundary

## Final permitted claims

- Round 10 is a controlled empirical characterization of S4, vanilla S3-S4,
  and S2-S4 whole-stage reuse on B0/B1 and one RTX 4090 D.
- The three boundaries show reproducible quality/latency frontiers. S4/FI-2 is
  the lowest-damage point for both models; S2-S4/FI-8 is the highest-speed point.
- S4-only is a distinct conservative operating point relative to vanilla S3-S4
  for both B0 and B1.
- The observed ordering is interpretable from measured stage runtime composition,
  but exact speedup is hardware- and implementation-dependent.
- S3-S4 is only a vanilla high-level feature-reuse baseline without alignment or
  fusion; it is not a reproduction of TFC.

## Prohibited claims

- Do not claim first temporal feature reuse, first stage reuse, first keyframe
  reuse, or a new S3-S4 caching method.
- Do not claim universal boundary optimality or hardware portability.
- Do not combine historical RTX 4090 timings with the Round-10 RTX 4090 D table.
- Do not report component speedup as model speedup; use paired end-to-end latency.
