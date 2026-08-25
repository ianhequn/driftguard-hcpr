# DriftGuard Round 7 Restart Progress

日期：2026-08-25

## 目标

在冻结的 VSPW validation split、VSPW-124 SegFormer-B0 和 whole-Stage4 temporal reuse 语义下，验证真实 ground-truth accuracy 与 CUDA-resident end-to-end latency 是否同时成立。

## 已完成阶段

1. **Preflight / artifact gate**：模型 SHA256 与冻结值一致；标签类别数为 124；Stage1-4 shape contract 通过。
2. **100% refresh equivalence**：20 个 validation frames 上 Full 与 StructuredReusePipeline 逐项一致，误差为零，prediction disagreement 为零。
3. **Full GT baseline**：24,392 frames / 343 videos，mIoU 与冻结参考值完全一致。
4. **FI-2 quality gate**：FI-2 mIoU drop 为 0.0919 pp，未触发 5 pp hard stop。
5. **Full curve**：完成 FI-3 / FI-4 / FI-6 / FI-8 的完整 validation benchmark。
6. **Latency / profile**：完成 200 warmup + 1000 measurements 的 CUDA Event paired latency，以及 Stage1-3 / Stage4 / Decoder profile。
7. **Analysis**：完成 video bootstrap、per-class comparison、Pareto figures 与决策文件。
8. **Verification**：云端项目测试 `94 passed, 2 skipped`；仅保留既有 pytest warnings，未出现失败。

## 冻结边界

- Stage1-3 每帧 fresh；Stage4 只允许 dense refresh 或复用 final F4；decoder 每帧执行。
- 每个 video 第一帧 refresh，video 间 cache reset，按时间顺序处理，不 shuffle。
- 只测试 Full、FI-2、FI-3、FI-4、FI-6、FI-8；FI-2 是预先指定的 primary operating point。
- 不使用 AFVT、VisualRisk、StageAgeVisual、sparse/partial computation、custom CUDA、Triton、TensorRT 或新模型/数据。

## 当前结论

`ROUND 7 RESTART = STRONG GO`

依据是 primary FI-2 同时满足真实配对端到端 speedup >= 1.10 和 mIoU drop <= 0.5 pp。完整证据见 `round7_restart_final_report.md`。
