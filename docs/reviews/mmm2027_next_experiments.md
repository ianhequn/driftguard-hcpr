# MMM 2027 下一轮最小实验设计

目标不是堆实验，而是用三个 kill tests 判断三个核心词是否还能保留：`Hardware-Calibrated`、`Progressive`、`Temporal Reuse`。所有正式实验必须在恢复冻结 B0/B1 checkpoint、匹配 CUDA 环境后执行；当前机器 CPU-only，暂不运行。

## 总协议冻结

- 数据：同一个 VSPW val split，343 videos / 24,392 frames，按 video natural order。
- 模型：B0、B1 同一 frozen `model.safetensors`，记录实际 SHA256。
- 输入：512×512，batch 1，BF16；Full/fixed/two-action/HCPR 共用预处理。
- 状态：视频边界 cache reset；第一帧 Full；每个 method 的 quality run 独立初始化，但使用完全相同的视频顺序。
- timing：CUDA-resident input 到 ready logits；包含 controller、scalar synchronization、cache bookkeeping、selected stages、decoder；200 warmup、1,000 CUDA events、5 repetitions、同进程 paired Full baseline。
- calibration：只用 32 个 train videos，seed 42，无 label；保存每个 boundary 的 scale、score aggregation、epsilon 和完整 lambda/threshold grid。
- confidence interval：以 video 为 resampling unit；speedup 同一 repetition 内与 paired Full 配对，报告 mean、95% CI 和 absolute latency。

## Kill Test A：device-matched fixed/adaptive Pareto

### 问题

“HCPR 形成更密的 quality-latency operating set”是否在同一设备、同一模型、同一 timing protocol 下仍成立？

### 方法

每个 GPU、每个模型都重新产生：

1. Full；
2. fixed boundary × refresh interval 完整网格：S4、S3-S4、S2-S4 × FI-1/2/3/4/6/8（资源不足时至少保留论文网格并说明删减）；
3. HCPR 完整 Q20/Q35/Q50/Q65/Q80/Q90/Q95。

每个点都保存：quality source、latency source、device、protocol ID、absolute latency、speedup、CI、mIoU damage、stage calls。Pareto 脚本只接受相同 `model/device/protocol_id` 的点。

### Kill 条件

- 若某一设备上 HCPR 的大部分点被 fixed frontier 支配，`Progressive` claim 降级为“提供一种 adaptive operating set”。
- 若 fixed quality 与 HCPR quality 仍来自不同 run，A 直接判定为 invalid，不进入论文。

## Kill Test B：Two-action vs Progressive

### 四个方法

1. `Full`：每帧完整执行。
2. `Fixed-S2-S4`：固定 boundary + refresh interval 网格。
3. `TwoAction`：仅在 `Full` 与 `S2-S4` 间选择。
4. `HCPR-Progressive`：Full、S2-S4、S3-S4、S4 四动作。

TwoAction 必须复用 HCPR 的：同一 temporal drift、同一 scale、同一 age guard、同一 reference update、同一视频顺序、同一 timing boundary；唯一变化是动作集合。阈值必须在独立无标签 calibration 上得到自己的 operating grid，不能直接沿用 progressive 的数值 lambda。

### 三种比较视角

- matched latency：在相同 latency 容差内比较 mIoU damage；
- matched damage：在相同 damage 容差内比较 latency/speedup；
- matched refresh budget：匹配 Stage 4 refresh ratio 与平均 stage calls/frame，再比较 mIoU 和 temporal metrics。

同时报告 S3-S4/S4 的 action fraction，以及移除中间动作后与 HCPR 的 per-frame route agreement。

### Kill 条件

- TwoAction 在三种匹配视角都不劣于 HCPR：撤回“progressive multi-boundary 的独立增量”，保留更简单的 adaptive controller。
- HCPR 只有在至少一个 matching 视角稳定优于 TwoAction，且优势不由极少数异常视频贡献时，才保留 `Progressive`。

## Kill Test C：Temporal consistency

### 必须新增的输出

- `predictions/{method}/{video_id}/{frame_id}.png` 或压缩的 uint8/uint16 mask shard；
- `frame_metrics.csv`：video、frame、mIoU/IoU degradation、route、cache age、refresh flags；
- `route_log.jsonl`：每帧 boundary score、chosen route、cache age、forced refresh；
- `temporal_metrics.json`：mVC8、mVC16、实现版本、clip/window 定义；
- 代表视频连续帧可视化：RGB、GT、Full、fixed、TwoAction、HCPR prediction、route/cache age。

### 比较点

- Full；
- fixed S2-S4；
- TwoAction conservative/middle/aggressive；
- HCPR conservative/middle/aggressive。

每个点同时报告 mIoU、mVC8、mVC16、absolute latency、speedup、per-frame degradation。按 low-motion、high-motion、scene-cut、small-object subset 分层；subset 规则必须在看结果前冻结。

### Kill 条件

- HCPR 的 mIoU 优势伴随明显 mVC8/mVC16 崩溃：不能称为稳定的视频 segmentation reuse。
- Full 与 reuse 的 temporal gap 只在高运动/scene-cut subset 出现：主 claim 必须改成条件性 claim，并强调 age guard/运动 failure mode。

## 必做 supporting ablations

### Age guard

`1 / 3 / 7 / 无 guard`。记录 maximum stale run、per-frame damage、mVC8/mVC16、route mix。无 guard 只能作为诊断，不得用于主 Pareto。

### Cost/threshold fairness

- `NewCost`：target-device measured k；
- `OldCost`：source-device k；
- `DriftOnly`：所有 k=1，但独立 calibration；
- `ConstantCost`：保留近似总成本尺度，去掉 stage 相对差异。

对每个 variant 输出完整 operating curve、matched latency/damage/refresh 三种表。禁止只按同名 Q 标签比较。

### Global vs local drift

保存 global GAP drift 与局部 patch/motion observable，按 per-frame degradation 做 calibration binning。目标不是凭结果换 controller，而是回答 global score 是否在 localized motion 下失效。

### Label protocol

在新实验 provenance 中记录：raw 0、raw 124、raw 253、raw 255 的 normalization；canonical valid range `0..123`；ignore `255`；官方 evaluator equivalence test。

## 预期新增文件

```text
driftguard/structured_reuse/two_action.py
driftguard/metrics/temporal_consistency.py
scripts/run_round12_fixed_quality.py
scripts/run_round12_two_action_quality.py
scripts/run_round12_temporal_quality.py
scripts/run_round12_collect_predictions.py
scripts/audit_round12_protocol.py
tests/test_two_action_controller.py
tests/test_temporal_consistency.py
tests/test_round12_protocol.py
results/R12_E00_preflight/
results/R12_E01_calibration/
results/R12_E02_fixed/
results/R12_E03_two_action/
results/R12_E04_progressive/
results/R12_E05_temporal/
results/R12_E06_analysis/
```

## 恢复 checkpoint 后的执行顺序

```text
1. run_round12_preflight.py：确认 B0/B1 SHA256、GPU、CUDA、数据和 label contract
2. audit_round12_protocol.py：冻结 protocol ID、顺序、cache reset、threshold grid
3. Kill Test A：先做 fixed/adaptive matched Pareto
4. Kill Test B：再做 TwoAction vs Progressive
5. Kill Test C：最后保存 masks 并计算 mVC8/mVC16
6. 只根据三个 kill tests 的结果决定论文标题和核心 claim
```

## 唯一最优先实验

**恢复 B0/B1 冻结 checkpoint 后，先执行 Kill Test A 的 device-matched fixed/adaptive Pareto。** 原因是当前最直接的 correctness blocker 是 fixed/HCPR 来源链；在它修正前，继续讨论 hardware calibration 或 progressive novelty 都没有可靠比较基准。
