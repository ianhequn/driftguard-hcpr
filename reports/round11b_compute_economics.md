# Round 11B Compute Economics

This round measures transfer on a second GPU with the frozen SegFormer-B0/B1 models. The cross-hardware claim uses paired CUDA-resident input-to-logits measurements with 200 warmup, 1000 measurements, and 5 repetitions.

- B0: Full profile 11.052737 ms; Stage 2/3/4 marginal profiles 2.603709/2.728337/2.288658 ms; k=0.235571,0.246847,0.207067.
- B1: Full profile 11.028257 ms; Stage 2/3/4 marginal profiles 2.745045/2.733099/2.246174 ms; k=0.248910,0.247827,0.203674.

Controller overhead is reported separately and is included in the paired latency boundary. Profile-only k transfer is not a speedup claim.
